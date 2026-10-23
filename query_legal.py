"""Interactive legal inquiry CLI supporting BM25/Hybrid retrieval, Ollama, and external LLM APIs."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request
import pyarrow.parquet as pq
import requests

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import BM25Retriever
from src.rag.query.normalizer import LegalAcronymNormalizer
from src.rag.query.contextualizer import ConversationContextualizer
from src.rag.output.formatter import LegalResponseFormatter
from src.rag.context.compressor import LegalPassageCompressor

DEFAULT_CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_retrieval"
    / "corpus_articles_preprocessed.parquet"
)
DEFAULT_BM25_INDEX_PATH = PROJECT_ROOT / "data" / "indices" / "bm25_index.pkl"
DEFAULT_DENSE_INDEX_DIR = PROJECT_ROOT / "data" / "indices" / "dense_bge_m3"


def check_ollama_running(host: str = "http://localhost:11434") -> bool:
    try:
        req = urllib.request.Request(f"{host}/api/tags", headers={"User-Agent": "LegalRAG"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def call_ollama(
    prompt: str,
    system_prompt: str,
    model: str = "qwen2.5:1.5b",
    host: str = "http://localhost:11434",
    stream_to_stdout: bool = True,
) -> str:
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": True,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "repeat_penalty": 1.15,
            "num_predict": 512,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    full_text = []
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    text = chunk.get("response", "")
                    if stream_to_stdout:
                        sys.stdout.write(text)
                        sys.stdout.flush()
                    full_text.append(text)
        if stream_to_stdout:
            print("\n")
    except Exception as err:
        if stream_to_stdout:
            print(f"\n[Lỗi kết nối Ollama]: {err}")
            print("Mẹo: Đảm bảo Ollama đang chạy và model đã được tải: 'ollama run qwen2.5:1.5b'")
    return "".join(full_text)


def fetch_available_models(api_key: str, api_base: str) -> List[str]:
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    if "/chat/completions" in api_base:
        models_url = api_base.replace("/chat/completions", "/models")
    else:
        models_url = "https://api.groq.com/openai/v1/models"
    try:
        resp = requests.get(models_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            return [m["id"] for m in data if "id" in m and not m["id"].startswith("whisper")]
    except Exception:
        pass
    return []


def call_groq_or_openai(
    prompt: str,
    system_prompt: str,
    api_key: str,
    model: str = "llama-3.3-70b-versatile",
    api_base: str = "https://api.groq.com/openai/v1/chat/completions",
    stream_to_stdout: bool = True,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    current_model = model
    payload = {
        "model": current_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "frequency_penalty": 0.4,
        "presence_penalty": 0.1,
        "max_tokens": 1024,
        "stream": True,
    }
    full_text = []
    try:
        response = requests.post(
            api_base,
            json=payload,
            headers=headers,
            stream=True,
            timeout=60,
        )
        if response.status_code == 404 or (response.status_code != 200 and "model_not_found" in response.text):
            available = fetch_available_models(api_key, api_base)
            if available:
                if stream_to_stdout:
                    print(f"   [!] Model '{current_model}' không có sẵn.")
                    print(f"   -> Các model khả dụng trên tài khoản của bạn: {', '.join(available)}")
                preferred_order = [
                    "llama-3.1-70b-versatile",
                    "llama-3.3-70b-specdec",
                    "llama3-70b-8192",
                    "llama-3.1-8b-instant",
                    "llama3-8b-8192",
                    "mixtral-8x7b-32768",
                    "gemma2-9b-it",
                ]
                fallback = None
                for cand in preferred_order:
                    if cand in available:
                        fallback = cand
                        break
                if not fallback:
                    fallback = available[0]
                if stream_to_stdout:
                    print(f"   -> Tự động chuyển sang sử dụng model: '{fallback}'...\n")
                current_model = fallback
                payload["model"] = current_model
                response = requests.post(
                    api_base,
                    json=payload,
                    headers=headers,
                    stream=True,
                    timeout=60,
                )

        if response.status_code != 200:
            if stream_to_stdout:
                print(f"\n[Lỗi kết nối API] HTTP {response.status_code}: {response.text}")
            return ""

        response.encoding = "utf-8"
        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8", errors="replace").strip()
            if line_str.startswith("data: ") and line_str != "data: [DONE]":
                raw_data = line_str[6:]
                try:
                    chunk = json.loads(raw_data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if stream_to_stdout:
                        sys.stdout.write(content)
                        sys.stdout.flush()
                    full_text.append(content)
                except json.JSONDecodeError:
                    pass
        if stream_to_stdout:
            print("\n")
    except Exception as err:
        if stream_to_stdout:
            print(f"\n[Lỗi kết nối API]: {err}")
    return "".join(full_text)


def call_gemini(
    prompt: str,
    system_prompt: str,
    api_key: str,
    model: str = "gemini-2.0-flash",
    stream_to_stdout: bool = True,
) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key.strip()}"
    full_prompt = f"{system_prompt}\n\n---\n{prompt}"
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1024},
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code != 200:
            if stream_to_stdout:
                print(f"\n[Lỗi kết nối Gemini API] HTTP {response.status_code}: {response.text}")
            return ""
        result = response.json()
        text = (
            result.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        if stream_to_stdout:
            print(text + "\n")
        return text
    except Exception as err:
        if stream_to_stdout:
            print(f"\n[Lỗi kết nối Gemini API]: {err}")
        return ""


def call_colab(
    prompt: str,
    system_prompt: str,
    endpoint_url: str,
    stream_to_stdout: bool = True,
) -> str:
    url = endpoint_url.strip().rstrip("/")
    if not url.endswith("/v1/chat/completions"):
        url = f"{url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    payload = {
        "model": "qwen2.5-7b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 512,
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=180)
        if response.status_code != 200:
            if stream_to_stdout:
                print(f"\n[Lỗi kết nối Colab API] HTTP {response.status_code}: {response.text}")
            return ""
        result = response.json()
        text = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if stream_to_stdout:
            print(text + "\n")
        return text
    except Exception as err:
        if stream_to_stdout:
            print(f"\n[Lỗi kết nối Colab]: {err}")
        return ""


class LegalAssistant:
    def __init__(
        self,
        retriever_type: str = "hybrid_rerank",
        top_k: int = 6,
        verbose: bool = True,
    ):
        self.verbose = verbose
        if self.verbose:
            print("=" * 70)
            print("KHỞI TẠO HỆ THỐNG TRUY XUẤT PHÁP LUẬT VIỆT NAM (LEGAL RAG)")
            print("=" * 70)
        self.retriever_type = retriever_type.lower()
        self.top_k = top_k
        self.bm25_retriever = None
        self.dense_retriever = None
        self.reranker = None

        if self.retriever_type in ("bm25", "hybrid", "hybrid_rerank"):
            if self.verbose:
                print("1. Đang nạp chỉ mục BM25 (59,636 điều luật)...")
            t0 = time.time()
            self.bm25_retriever = BM25Retriever.load(DEFAULT_BM25_INDEX_PATH)
            if self.verbose:
                print(f"   -> Đã nạp chỉ mục BM25 trong {time.time() - t0:.2f} giây.")

        if self.retriever_type in ("dense", "hybrid", "hybrid_rerank"):
            if self.verbose:
                print("2. Đang nạp chỉ mục Dense BGE-M3 (FAISS Index)...")
            t0 = time.time()
            from src.retrieval.dense.encoder import DenseEncoder
            from src.retrieval.dense.retriever import DenseRetriever

            self.encoder = DenseEncoder()
            self.dense_retriever = DenseRetriever.load(DEFAULT_DENSE_INDEX_DIR, encoder=self.encoder)
            if self.verbose:
                print(f"   -> Đã nạp chỉ mục Dense trong {time.time() - t0:.2f} giây.")

        if self.retriever_type == "hybrid_rerank":
            if self.verbose:
                print("3. Đang nạp mô hình Reranker (BAAI/bge-reranker-base)...")
            t0 = time.time()
            from src.retrieval.reranking.cross_encoder import ArticleReranker

            self.reranker = ArticleReranker(device="cpu", max_length=384)
            if self.verbose:
                print(f"   -> Đã nạp Reranker trong {time.time() - t0:.2f} giây.")

        if self.verbose:
            print("4. Đang nạp dữ liệu điều luật...")
        t1 = time.time()
        table = pq.read_table(
            DEFAULT_CORPUS_PATH,
            columns=["global_article_id", "law_id", "aid", "text"],
        )
        df = table.to_pandas()
        self.corpus: Dict[int, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            self.corpus[int(row["global_article_id"])] = {
                "law_id": row["law_id"],
                "aid": row["aid"],
                "text": row["text"],
            }
        if self.verbose:
            print(f"   -> Đã nạp {len(self.corpus):,} điều luật trong {time.time() - t1:.2f} giây.")

        self.normalizer = LegalAcronymNormalizer()
        self.contextualizer = ConversationContextualizer()
        self.compressor = LegalPassageCompressor()
        self.history: List[Dict[str, str]] = []

    def _search_hybrid(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        bm25_res = self.bm25_retriever.search(query, top_k=top_k * 2) if self.bm25_retriever else []
        dense_res = self.dense_retriever.search(query, top_k=top_k * 2) if self.dense_retriever else []

        rrf_scores: Dict[int, float] = {}
        for rank, r in enumerate(bm25_res, start=1):
            rrf_scores[r.article_id] = rrf_scores.get(r.article_id, 0.0) + (1.0 / (60 + rank))
        for rank, r in enumerate(dense_res, start=1):
            rrf_scores[r.article_id] = rrf_scores.get(r.article_id, 0.0) + (1.0 / (60 + rank))

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        docs = []
        for rank, aid in enumerate(sorted_ids, start=1):
            doc_info = self.corpus.get(aid, {})
            docs.append(
                {
                    "rank": rank,
                    "article_id": aid,
                    "score": float(rrf_scores[aid]),
                    "law_id": doc_info.get("law_id", ""),
                    "aid": doc_info.get("aid", ""),
                    "text": doc_info.get("text", ""),
                }
            )
        return docs

    def search(self, query: str) -> List[Dict[str, Any]]:
        if self.retriever_type == "hybrid_rerank":
            pool_size = max(15, self.top_k * 2)
            candidates = self._search_hybrid(query, top_k=pool_size)
            if not self.reranker or not candidates:
                return candidates[: self.top_k]

            candidate_ids = [c["article_id"] for c in candidates]
            candidate_texts = [c["text"] for c in candidates]
            rerank_results = self.reranker.rerank(
                query=query,
                candidate_ids=candidate_ids,
                candidate_texts=candidate_texts,
                top_k=self.top_k,
            )
            docs = []
            for r in rerank_results:
                doc_info = self.corpus.get(r.article_id, {})
                docs.append(
                    {
                        "rank": r.rank,
                        "article_id": r.article_id,
                        "score": float(r.score),
                        "law_id": doc_info.get("law_id", ""),
                        "aid": doc_info.get("aid", ""),
                        "text": doc_info.get("text", ""),
                    }
                )
            return docs

        elif self.retriever_type == "hybrid":
            return self._search_hybrid(query, top_k=self.top_k)

        elif self.retriever_type == "dense":
            results = self.dense_retriever.search(query, top_k=self.top_k) if self.dense_retriever else []
        else:
            results = self.bm25_retriever.search(query, top_k=self.top_k) if self.bm25_retriever else []

        retrieved_docs = []
        for r in results:
            doc_info = self.corpus.get(r.article_id, {})
            retrieved_docs.append(
                {
                    "rank": r.rank,
                    "article_id": r.article_id,
                    "score": r.score,
                    "law_id": doc_info.get("law_id", ""),
                    "aid": doc_info.get("aid", ""),
                    "text": doc_info.get("text", ""),
                }
            )
        return retrieved_docs

    def build_context_and_prompt(
        self, query: str, docs: List[Dict[str, Any]]
    ) -> tuple[str, str]:
        system_prompt = LegalResponseFormatter.SYSTEM_PROMPT_TEMPLATE

        top_docs = self.compressor.compress_documents(query, docs[:3])
        context_blocks = []
        for d in top_docs:
            snippet = d["text"].strip()
            law_meta = f"Văn bản ID: {d['law_id']} | Điều ID: {d['aid']}" if d.get('law_id') else f"Article #{d.get('article_id')}"
            context_blocks.append(
                f"[TÀI LIỆU #{d['rank']} - {law_meta}]:\n{snippet}"
            )

        context_text = "\n\n".join(context_blocks)
        user_prompt = (
            f"=== CĂN CỨ PHÁP LÝ ĐƯỢC TRÍCH XUẤT TỪ DỮ LIỆU ===\n\n"
            f"{context_text}\n\n"
            f"=== CÂU HỎI CỦA NGƯỜI DÙNG ===\n"
            f"{query}\n\n"
            f"HÃY TRẢ LỜI CÂU HỎI TRÊN DỰA TRÊN CĂN CỨ PHÁP LÝ ĐƯỢC CUNG CẤP THEO ĐÚNG CẤU TRÚC 3 PHẦN:"
        )
        return user_prompt, system_prompt

    def answer_query(
        self,
        query: str,
        backend: str = "auto",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        colab_url: Optional[str] = None,
        as_json: bool = False,
    ) -> Dict[str, Any]:
        if not as_json:
            print("\n" + "-" * 70)
            print(f"🔍 CÂU HỎI: {query}")
            print("-" * 70)

        # 0.1 Contextualize follow-up queries
        ctx_query, is_follow_up, prior_topic = self.contextualizer.contextualize(query, self.history)
        if not as_json and is_follow_up and prior_topic:
            print(f"🔄 [Ngữ cảnh đa lượt]: Tiếp nối chủ đề: '{prior_topic}'")

        # 0.2 Normalize acronyms
        norm_query, detected_acronyms, _ = self.normalizer.normalize(ctx_query)
        if not as_json and detected_acronyms:
            exp_str = ", ".join([f"{k} -> {v}" for k, v in detected_acronyms.items()])
            print(f"💡 [Tiền xử lý Phase 13]: Nhận diện & chuẩn hóa từ viết tắt: {exp_str}")

        # 1. RETRIEVAL (Chạy máy local)
        t_start = time.time()
        docs = self.search(norm_query)
        search_time = (time.time() - t_start) * 1000

        # Lưu vào lịch sử hội thoại
        self.history.append({"role": "user", "content": query})

        if not as_json:
            print(f"⚡ [BƯỚC 1 - THỦ THƯ TÌM KIẾM]: Tìm thấy {len(docs)} điều luật phù hợp ({search_time:.1f}ms):")
            for d in docs:
                meta = f"Văn bản: {d['law_id']} | Điều: {d['aid']}" if d.get('law_id') else f"ID: {d['article_id']}"
                snippet = d["text"].replace("\n", " ")[:150] + "..." if len(d["text"]) > 150 else d["text"]
                print(f"  * [Top {d['rank']}] (Score: {d['score']:.2f}) [{meta}]")
                print(f"    Trích đoạn: {snippet}")

        # 1.5 Nén trích đoạn điều luật dài (Phase 15: Passage Compression)
        compressed_top = self.compressor.compress_documents(ctx_query, docs[:3])
        comp_summary = [
            f"Top {d['rank']} (giữ {len(d.get('clauses_retained', []))} phần, nén {int((1.0 - d.get('compression_ratio', 1.0)) * 100)}%)"
            for d in compressed_top
            if d.get("is_compressed")
        ]
        if not as_json and comp_summary:
            print(f"✂️ [Nén ngữ cảnh Phase 15]: Đã lọc & nén khoản trọng tâm cho: {', '.join(comp_summary)}")

        # 2. XÂY DỰNG PROMPT
        user_prompt, system_prompt = self.build_context_and_prompt(ctx_query, docs)

        # 3. XÁC ĐỊNH BACKEND LLM
        effective_backend = backend
        if backend == "auto":
            if colab_url or os.environ.get("COLAB_URL"):
                effective_backend = "colab"
            elif api_key or os.environ.get("GROQ_API_KEY"):
                effective_backend = "groq"
            elif os.environ.get("GEMINI_API_KEY"):
                effective_backend = "gemini"
            elif os.environ.get("OPENAI_API_KEY"):
                effective_backend = "openai"
            elif check_ollama_running():
                effective_backend = "ollama"
            else:
                effective_backend = "none"

        if not as_json:
            print(f"\n🤖 [BƯỚC 2 - TỔNG HỢP CÂU TRẢ LỜI] (Chế độ: {effective_backend.upper()}):")

        full_response = ""
        if effective_backend == "colab":
            endpoint = colab_url or os.environ.get("COLAB_URL")
            if not endpoint:
                if as_json:
                    err_payload = {"status": "ERROR", "error": "--colab-url is required"}
                    print(json.dumps(err_payload, ensure_ascii=False, indent=2))
                    return err_payload
                print("   [!] Chưa truyền --colab-url. Vui lòng truyền URL Cloudflare Tunnel.")
                return {}
            if not as_json:
                print(f"   -> Đang gửi ngữ cảnh luật tới GPU Colab ({endpoint})...\n")
                print("📝 TRẢ LỜI:\n")
            full_response = call_colab(
                user_prompt,
                system_prompt,
                endpoint_url=endpoint,
                stream_to_stdout=(not as_json),
            )

        elif effective_backend == "ollama":
            active_model = model or "qwen2.5:1.5b"
            if not as_json:
                print(f"   -> Đang gửi ngữ cảnh luật tới Ollama (Model: {active_model})...\n")
                print("📝 TRẢ LỜI:\n")
            full_response = call_ollama(
                user_prompt,
                system_prompt,
                model=active_model,
                stream_to_stdout=(not as_json),
            )

        elif effective_backend == "groq":
            key = api_key or os.environ.get("GROQ_API_KEY")
            active_model = model or "llama-3.3-70b-versatile"
            if not key:
                if as_json:
                    err_payload = {"status": "ERROR", "error": "GROQ_API_KEY is required"}
                    print(json.dumps(err_payload, ensure_ascii=False, indent=2))
                    return err_payload
                print("   [!] Chưa tìm thấy GROQ_API_KEY. Vui lòng truyền --api-key hoặc set biến môi trường.")
                return {}
            if not as_json:
                print(f"   -> Đang gửi ngữ cảnh luật tới Groq Cloud (Model: {active_model})...\n")
                print("📝 TRẢ LỜI:\n")
            full_response = call_groq_or_openai(
                user_prompt,
                system_prompt,
                api_key=key,
                model=active_model,
                api_base="https://api.groq.com/openai/v1/chat/completions",
                stream_to_stdout=(not as_json),
            )

        elif effective_backend == "gemini":
            key = api_key or os.environ.get("GEMINI_API_KEY")
            active_model = model or "gemini-2.0-flash"
            if not key:
                if as_json:
                    err_payload = {"status": "ERROR", "error": "GEMINI_API_KEY is required"}
                    print(json.dumps(err_payload, ensure_ascii=False, indent=2))
                    return err_payload
                print("   [!] Chưa tìm thấy GEMINI_API_KEY. Vui lòng truyền --api-key hoặc set biến môi trường.")
                return {}
            if not as_json:
                print(f"   -> Đang gửi ngữ cảnh luật tới Google Gemini (Model: {active_model})...\n")
                print("📝 TRẢ LỜI:\n")
            full_response = call_gemini(
                user_prompt,
                system_prompt,
                api_key=key,
                model=active_model,
                stream_to_stdout=(not as_json),
            )

        else:
            if not as_json:
                print("   (Chế độ Retrieval-Only: Không có LLM nào được kích hoạt)")
                print("\n💡 ĐÂY LÀ ĐOẠN VĂN BẢN ĐÃ ĐƯỢC BỌC TRONG PROMPT CHUẨN BỊ GỬI CHO LLM:")
                print("=" * 60)
                print(user_prompt[:600] + "\n... [các điều luật khác] ...")
                print("=" * 60)
                print("\n💡 Hướng dẫn chạy kèm LLM:")
                print("  1. Dùng Colab GPU (Qwen2.5-7B): python query_legal.py --backend colab --colab-url https://xxxx.trycloudflare.com")
                print("  2. Dùng Groq API (Llama 3.3 70B): python query_legal.py --backend groq --api-key gsk_...")
                print("  3. Dùng Ollama Local: python query_legal.py --backend ollama")

        api_json = LegalResponseFormatter.to_api_json(full_response, docs)
        if not as_json and full_response.strip() and effective_backend != "none":
            grounding = api_json.get("grounding", {})
            badge = grounding.get("badge", "")
            summary = grounding.get("summary", "")
            if badge:
                print("-" * 70)
                print(f"🛡️ [KIỂM CHỨNG CĂN CỨ & TRÍCH DẪN - PHASE 15]: {badge}")
                if summary:
                    print(f"   -> {summary}")
                print("-" * 70)

        if as_json:
            print(json.dumps(api_json, ensure_ascii=False, indent=2))
        return api_json


def main():
    parser = argparse.ArgumentParser(
        description="Tra cứu và Hỏi đáp Pháp luật Việt Nam (Legal RAG Interactive)"
    )
    parser.add_argument("--query", type=str, default=None, help="Câu hỏi trực tiếp (nếu không truyền sẽ vào REPL)")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "colab", "ollama", "groq", "gemini", "openai", "none"],
        help="Backend LLM (mặc định: auto - tự nhận diện)",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="hybrid_rerank",
        choices=["hybrid_rerank", "hybrid", "dense", "bm25"],
        help="Loại retriever: 'hybrid_rerank' (chuẩn xác nhất), 'hybrid' (BM25 + Dense RRF), 'dense', hoặc 'bm25'",
    )
    parser.add_argument("--model", type=str, default=None, help="Tên model cụ thể")
    parser.add_argument("--api-key", type=str, default=None, help="API key của Groq/Gemini/OpenAI")
    parser.add_argument("--colab-url", type=str, default=None, help="Cloudflare Tunnel URL kết nối Colab GPU (ví dụ: https://xxxx.trycloudflare.com)")
    parser.add_argument("--top-k", type=int, default=6, help="Số điều luật trích xuất (mặc định: 6)")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Xuất kết quả ở định dạng JSON API có cấu trúc (Phase 16: Structured Output)",
    )
    args = parser.parse_args()

    assistant = LegalAssistant(
        retriever_type=args.retriever,
        top_k=args.top_k,
        verbose=(not args.json),
    )

    if args.query:
        assistant.answer_query(
            args.query,
            backend=args.backend,
            model=args.model,
            api_key=args.api_key,
            colab_url=args.colab_url,
            as_json=args.json,
        )
    else:
        print("\n" + "=" * 70)
        print("SẴN SÀNG! Bạn có thể gõ câu hỏi pháp lý bất kỳ (Gõ 'q' hoặc 'exit' để thoát).")
        print("=" * 70)
        while True:
            try:
                user_q = input("\n👉 Nhập câu hỏi pháp luật: ").strip()
                if not user_q:
                    continue
                if user_q.lower() in ["q", "exit", "quit"]:
                    print("Tạm biệt!")
                    break
                assistant.answer_query(
                    user_q,
                    backend=args.backend,
                    model=args.model,
                    api_key=args.api_key,
                    colab_url=args.colab_url,
                    as_json=args.json,
                )
            except (KeyboardInterrupt, EOFError):
                print("\nTạm biệt!")
                break


if __name__ == "__main__":
    main()
