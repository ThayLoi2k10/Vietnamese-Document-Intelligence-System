"""Vietnamese Legal RAG Assistant - Streamlit Web Application (Phase 18 UI)."""

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Generator, List, Optional
import pandas as pd
import pyarrow.parquet as pq
import requests
import streamlit as st

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

HF_BASE_URL = "https://huggingface.co/spaces/ThayLoi2k10/vietnamese-legal-rag/resolve/main"

def _ensure_remote_file(path: Path, rel_url: str, min_size: int = 1_000_000):
    if path.exists() and path.stat().st_size >= min_size:
        return
    if path.exists():
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF_BASE_URL}/{rel_url}"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        import urllib.request
        with st.spinner(f"Đang đồng bộ dữ liệu ({path.name})... Quá trình này chỉ mất 15-30 giây."):
            urllib.request.urlretrieve(url, tmp_path)
            if tmp_path.exists() and tmp_path.stat().st_size >= min_size:
                tmp_path.replace(path)
    except Exception as e:
        st.error(f"Lỗi khi tải file {path.name}: {e}")


@st.cache_resource(show_spinner="Đang nạp cơ sở dữ liệu luật (59,636 điều)...")
def load_corpus() -> Dict[int, Dict[str, Any]]:
    _ensure_remote_file(DEFAULT_CORPUS_PATH, "data/processed/legal_retrieval/corpus_articles_preprocessed.parquet", min_size=50_000_000)
    corpus: Dict[int, Dict[str, Any]] = {}
    if DEFAULT_CORPUS_PATH.exists():
        try:
            table = pq.read_table(
                DEFAULT_CORPUS_PATH,
                columns=["global_article_id", "law_id", "aid", "text"],
            )
            df = table.to_pandas()
            for _, row in df.iterrows():
                gid = int(row["global_article_id"])
                corpus[gid] = {
                    "law_id": str(row["law_id"]),
                    "aid": str(row["aid"]),
                    "text": str(row["text"]),
                }
        except Exception:
            try:
                DEFAULT_CORPUS_PATH.unlink(missing_ok=True)
            except Exception:
                pass
    return corpus


@st.cache_resource(show_spinner="Đang xây dựng danh mục văn bản luật...")
def get_laws_catalog(corpus: Dict[int, Dict[str, Any]]) -> Dict[str, List[int]]:
    catalog: Dict[str, List[int]] = {}
    for aid, doc in corpus.items():
        lid = doc.get("law_id") or "Văn bản khác"
        if lid not in catalog:
            catalog[lid] = []
        catalog[lid].append(aid)
    return catalog


@st.cache_resource(show_spinner="Đang nạp chỉ mục từ khóa BM25...")
def load_bm25():
    _ensure_remote_file(DEFAULT_BM25_INDEX_PATH, "data/indices/bm25_index.pkl", min_size=20_000_000)
    if DEFAULT_BM25_INDEX_PATH.exists():
        return BM25Retriever.load(DEFAULT_BM25_INDEX_PATH)
    return None


@st.cache_resource(show_spinner="Đang nạp chỉ mục ngữ nghĩa Dense (BGE-M3 + FAISS)...")
def load_dense():
    if DEFAULT_DENSE_INDEX_DIR.exists():
        from src.retrieval.dense.encoder import DenseEncoder
        from src.retrieval.dense.retriever import DenseRetriever

        encoder = DenseEncoder()
        return DenseRetriever.load(DEFAULT_DENSE_INDEX_DIR, encoder=encoder)
    return None


@st.cache_resource(show_spinner="Đang nạp mô hình Reranker (BAAI/bge-reranker-base)...")
def load_reranker():
    try:
        import FlagEmbedding
        from src.retrieval.reranking.cross_encoder import ArticleReranker

        return ArticleReranker(device="cpu", max_length=384)
    except Exception:
        return None


@st.cache_resource
def load_normalizer():
    return LegalAcronymNormalizer()


@st.cache_resource
def load_contextualizer():
    return ConversationContextualizer()


@st.cache_resource
def load_compressor():
    return LegalPassageCompressor()


def search_bm25(
    query: str, top_k: int, bm25_retriever, corpus: Dict[int, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not bm25_retriever:
        return []
    results = bm25_retriever.search(query, top_k=top_k)
    docs = []
    for r in results:
        meta = corpus.get(r.article_id, {})
        docs.append(
            {
                "rank": r.rank,
                "article_id": r.article_id,
                "score": float(r.score),
                "law_id": meta.get("law_id", "N/A"),
                "aid": meta.get("aid", "N/A"),
                "text": meta.get("text", ""),
            }
        )
    return docs


def search_dense(
    query: str, top_k: int, dense_retriever, corpus: Dict[int, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not dense_retriever:
        return []
    results = dense_retriever.search(query, top_k=top_k)
    docs = []
    for r in results:
        meta = corpus.get(r.article_id, {})
        docs.append(
            {
                "rank": r.rank,
                "article_id": r.article_id,
                "score": float(r.score),
                "law_id": meta.get("law_id", "N/A"),
                "aid": meta.get("aid", "N/A"),
                "text": meta.get("text", ""),
            }
        )
    return docs


def search_hybrid(
    query: str,
    top_k: int,
    bm25_retriever,
    dense_retriever,
    corpus: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not dense_retriever and bm25_retriever:
        return search_bm25(query, top_k, bm25_retriever, corpus)
    bm25_res = bm25_retriever.search(query, top_k=top_k * 2) if bm25_retriever else []
    dense_res = dense_retriever.search(query, top_k=top_k * 2) if dense_retriever else []

    rrf_scores: Dict[int, float] = {}
    for rank, r in enumerate(bm25_res, start=1):
        rrf_scores[r.article_id] = rrf_scores.get(r.article_id, 0.0) + (1.0 / (60 + rank))
    for rank, r in enumerate(dense_res, start=1):
        rrf_scores[r.article_id] = rrf_scores.get(r.article_id, 0.0) + (1.0 / (60 + rank))

    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
    docs = []
    for rank, aid in enumerate(sorted_ids, start=1):
        meta = corpus.get(aid, {})
        docs.append(
            {
                "rank": rank,
                "article_id": aid,
                "score": float(rrf_scores[aid]),
                "law_id": meta.get("law_id", "N/A"),
                "aid": meta.get("aid", "N/A"),
                "text": meta.get("text", ""),
            }
        )
    return docs


def search_hybrid_rerank(
    query: str,
    top_k: int,
    bm25_retriever,
    dense_retriever,
    reranker,
    corpus: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pool_size = max(15, top_k * 2)
    candidates = search_hybrid(query, pool_size, bm25_retriever, dense_retriever, corpus)
    if not reranker or not candidates:
        return candidates[:top_k]

    candidate_ids = [c["article_id"] for c in candidates]
    candidate_texts = [c["text"] for c in candidates]
    try:
        rerank_results = reranker.rerank(
            query=query,
            candidate_ids=candidate_ids,
            candidate_texts=candidate_texts,
            top_k=top_k,
        )
    except Exception:
        return candidates[:top_k]
    docs = []
    for r in rerank_results:
        meta = corpus.get(r.article_id, {})
        docs.append(
            {
                "rank": r.rank,
                "article_id": r.article_id,
                "score": float(r.score),
                "law_id": meta.get("law_id", "N/A"),
                "aid": meta.get("aid", "N/A"),
                "text": meta.get("text", ""),
            }
        )
    return docs


def build_prompts(
    query: str, docs: List[Dict[str, Any]], compressor: Optional[LegalPassageCompressor] = None
) -> tuple[str, str, List[Dict[str, Any]]]:
    system_prompt = LegalResponseFormatter.SYSTEM_PROMPT_TEMPLATE

    active_docs = docs[:3]
    if compressor:
        compressed_docs = compressor.compress_documents(query, active_docs)
    else:
        compressed_docs = active_docs

    context_blocks = []
    for d in compressed_docs:
        snippet = d["text"].strip()
        law_meta = (
            f"Văn bản ID: {d['law_id']} | Điều ID: {d['aid']}"
            if d.get("law_id") != "N/A"
            else f"Article #{d.get('article_id')}"
        )
        context_blocks.append(f"[TÀI LIỆU #{d['rank']} - {law_meta}]:\n{snippet}")

    context_text = "\n\n".join(context_blocks)
    user_prompt = (
        f"=== CĂN CỨ PHÁP LÝ ĐƯỢC TRÍCH XUẤT TỪ DỮ LIỆU ===\n\n"
        f"{context_text}\n\n"
        f"=== CÂU HỎI CỦA NGƯỜI DÙNG ===\n"
        f"{query}\n\n"
        f"HÃY TRẢ LỜI CÂU HỎI TRÊN DỰA TRÊN CĂN CỨ PHÁP LÝ ĐƯỢC CUNG CẤP THEO ĐÚNG CẤU TRÚC 3 PHẦN:"
    )
    return user_prompt, system_prompt, compressed_docs


def fetch_groq_models(api_key: str) -> List[str]:
    verified_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]
    if not api_key:
        return verified_models
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "User-Agent": "Mozilla/5.0",
    }
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models", headers=headers, timeout=5
        )
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            excluded = ["whisper", "guard", "tts", "embed", "vision", "specdec"]
            models = [
                m["id"]
                for m in data
                if "id" in m and not any(kw in m["id"].lower() for kw in excluded)
            ]
            if models:
                # Prioritize high quality conversational models
                priority = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
                for p in reversed(priority):
                    if p in models:
                        models.remove(p)
                        models.insert(0, p)
                return models
    except Exception:
        pass
    return verified_models


def stream_groq(
    prompt: str, system_prompt: str, api_key: str, model: str
) -> Generator[str, None, None]:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
        "stream": True,
    }
    try:
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)
        # Self-healing: if model has a strict max_tokens limit (e.g. 512), retry automatically
        if response.status_code == 400 and "max_tokens" in response.text and payload["max_tokens"] > 512:
            payload["max_tokens"] = 512
            response = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)
        
        if response.status_code != 200:
            yield f"\n[Lỗi kết nối Groq API] HTTP {response.status_code}: {response.text}"
            return
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
                    if content:
                        yield content
                except json.JSONDecodeError:
                    pass
    except Exception as err:
        yield f"\n[Lỗi kết nối Groq]: {err}"


def stream_gemini(
    prompt: str, system_prompt: str, api_key: str, model: str = "gemini-1.5-flash"
) -> Generator[str, None, None]:
    clean_key = api_key.strip()
    full_prompt = f"{system_prompt}\n\n---\n{prompt}"
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
    }
    headers = {"Content-Type": "application/json"}

    candidate_urls = [
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse&key={clean_key}",
        f"https://generativelanguage.googleapis.com/v1/models/{model}:streamGenerateContent?alt=sse&key={clean_key}",
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}-latest:streamGenerateContent?alt=sse&key={clean_key}",
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse&key={clean_key}",
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={clean_key}",
        f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={clean_key}",
    ]

    last_error = ""
    for url in candidate_urls:
        try:
            is_stream = ":streamGenerateContent" in url
            response = requests.post(url, json=payload, headers=headers, stream=is_stream, timeout=45)
            if response.status_code == 200:
                if is_stream:
                    response.encoding = "utf-8"
                    for raw_line in response.iter_lines():
                        if not raw_line:
                            continue
                        line_str = raw_line.decode("utf-8", errors="replace").strip()
                        if line_str.startswith("data:"):
                            line_str = line_str[5:].strip()
                        if '"text":' in line_str:
                            import json
                            try:
                                chunk_json = json.loads(line_str)
                                text = chunk_json["candidates"][0]["content"]["parts"][0]["text"]
                                yield text
                            except Exception:
                                pass
                    return
                else:
                    data = response.json()
                    yield data["candidates"][0]["content"]["parts"][0]["text"]
                    return
            else:
                last_error = f"HTTP {response.status_code}: {response.text}"
        except Exception as e:
            last_error = str(e)

    yield f"\n[Lỗi kết nối Gemini API]: {last_error}\n"


def stream_colab(
    prompt: str, system_prompt: str, endpoint_url: str
) -> Generator[str, None, None]:
    url = endpoint_url.strip().rstrip("/")
    if not url.endswith("/v1/chat/completions"):
        url = f"{url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "qwen2.5-7b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=180)
        if response.status_code != 200:
            yield f"\n[Lỗi kết nối Colab API] HTTP {response.status_code}: {response.text}"
            return
        result = response.json()
        text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        yield text
    except Exception as err:
        yield f"\n[Lỗi kết nối Colab]: {err}"


def stream_ollama(
    prompt: str, system_prompt: str, model: str = "qwen2.5:1.5b"
) -> Generator[str, None, None]:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": True,
        "options": {"temperature": 0.2, "repeat_penalty": 1.15, "num_predict": 512},
    }
    try:
        response = requests.post(url, json=payload, stream=True, timeout=60)
        if response.status_code != 200:
            yield f"\n[Lỗi kết nối Ollama] HTTP {response.status_code}: {response.text}"
            return
        response.encoding = "utf-8"
        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8", errors="replace").strip()
            try:
                chunk = json.loads(line_str)
                text = chunk.get("response", "")
                if text:
                    yield text
            except json.JSONDecodeError:
                pass
    except Exception as err:
        yield f"\n[Lỗi kết nối Ollama]: {err}"


def main():
    st.set_page_config(
        page_title="Hệ Thống Trợ Lý Pháp Luật Việt Nam (Legal RAG)",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.1rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.0rem;
            color: #4B5563;
            margin-bottom: 1.2rem;
        }
        .law-badge {
            display: inline-block;
            background-color: #E0E7FF;
            color: #3730A3;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 600;
            margin-right: 6px;
        }
        .score-badge {
            display: inline-block;
            background-color: #ECFDF5;
            color: #065F46;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .result-card {
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 12px;
            background-color: #F9FAFB;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Nạp dữ liệu nền tảng
    corpus = load_corpus()
    laws_catalog = get_laws_catalog(corpus)
    bm25_retriever = load_bm25()
    normalizer = load_normalizer()
    contextualizer = load_contextualizer()
    compressor = load_compressor()

    # --- SIDEBAR CONFIGURATION ---
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/scales.png", width=64)
        st.title("⚖️ Cấu Hình RAG")

        st.subheader("1. Phương Pháp Truy Xuất")
        retriever_choice = st.selectbox(
            "Retriever:",
            [
                "BM25 (Toàn bộ 59,636 điều luật - Ổn định & Tối ưu Cloud)",
                "Hybrid + Reranker (Yêu cầu GPU Server)",
                "Hybrid (BM25 + Dense RRF)",
                "Dense (BGE-M3 Semantic)",
            ],
            index=0,
            help="BM25 quét tức thì trên toàn bộ 59,636 điều luật Việt Nam với độ trễ < 10ms, tối ưu tuyệt đối cho môi trường Cloud.",
        )

        top_k = st.slider("Số điều luật trích xuất (Top-K):", min_value=1, max_value=10, value=3)

        st.divider()

        st.subheader("2. Mô hình Ngôn ngữ (LLM)")
        backend_choice = st.selectbox(
            "Backend trả lời:",
            [
                "Google Gemini API (Khuyên dùng)",
                "Chỉ tra cứu điều luật (Không dùng LLM)",
                "Groq Cloud API (Llama 3.3 70B)",
                "Google Colab GPU Server (Qwen-7B)",
                "Ollama Local (Offline)",
            ],
            index=0,
        )

        colab_url = ""
        if "Colab" in backend_choice:
            colab_url = st.text_input(
                "Colab Tunnel URL:",
                value="",
                placeholder="https://xxxx.trycloudflare.com",
                help="Dán URL do Cloudflare Tunnel sinh ra ở bước chạy Colab.",
            )

        groq_api_key = ""
        groq_model = "llama-3.1-8b-instant"
        if "Groq" in backend_choice:
            default_key = os.environ.get("GROQ_API_KEY", "")
            groq_api_key = st.text_input(
                "Groq API Key:",
                value=default_key,
                type="password",
                help="Lấy API Key tại console.groq.com",
            )
            available_models = fetch_groq_models(groq_api_key)
            default_idx = 0
            if "llama-3.3-70b-versatile" in available_models:
                default_idx = available_models.index("llama-3.3-70b-versatile")
            elif "llama-3.1-8b-instant" in available_models:
                default_idx = available_models.index("llama-3.1-8b-instant")
            groq_model = st.selectbox("Model Groq:", available_models, index=default_idx)

        gemini_api_key = ""
        gemini_model = "gemini-1.5-flash"
        if "Gemini" in backend_choice:
            default_gemini_key = os.environ.get("GEMINI_API_KEY", "")
            gemini_api_key = st.text_input("Gemini API Key:", value=default_gemini_key, type="password")
            gemini_model = st.selectbox("Model Gemini:", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)

        ollama_model = "qwen2.5:1.5b"
        if "Ollama" in backend_choice:
            ollama_model = st.text_input("Tên Model Ollama:", value="qwen2.5:1.5b")

        st.divider()
        st.caption(f"📊 Dữ liệu: **{len(corpus):,} điều luật** | **{len(laws_catalog):,} văn bản**")
        if st.button("🗑️ Xóa lịch sử hỏi đáp", use_container_width=True):
            st.session_state.messages = []
            st.session_state.query_logs = []
            st.rerun()

    dense_retriever = None
    if any(x in retriever_choice for x in ["Dense", "Hybrid"]):
        dense_retriever = load_dense()

    reranker = None
    if "Reranker" in retriever_choice:
        reranker = load_reranker()

    # Session states
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "query_logs" not in st.session_state:
        st.session_state.query_logs = []

    # --- HEADER ---
    st.markdown('<div class="main-header">⚖️ Hệ Thống Trợ Lý Pháp Luật Việt Nam (Legal RAG)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Hệ thống AI tra cứu chính xác từ <b>59,636 điều luật</b> và trả lời có căn cứ pháp lý minh bạch.</div>',
        unsafe_allow_html=True,
    )

    # --- TABS GIAO DIỆN (PHASE 18) ---
    tab_chat, tab_search, tab_docs, tab_history = st.tabs(
        [
            "💬 Hỏi Đáp Pháp Lý (Chatbot)",
            "🔍 Tra Cứu Điều Luật (Fast Search)",
            "📚 Thư Viện Văn Bản (Document Explorer)",
            "📊 Lịch Sử & Thống Kê (Analytics)",
        ]
    )

    # =========================================================================
    # TAB 1: HỎI ĐÁP PHÁP LÝ (LEGAL CHAT)
    # =========================================================================
    with tab_chat:
        st.markdown("**💡 Câu hỏi gợi ý bạn có thể thử ngay:**")
        col1, col2, col3 = st.columns(3)
        sample_q = None
        if col1.button("📋 NLĐ đơn phương chấm dứt HĐLĐ phải báo trước bao nhiêu ngày?"):
            sample_q = "Người lao động đơn phương chấm dứt hợp đồng lao động phải báo trước bao nhiêu ngày"
        if col2.button("⏳ Thời gian thử việc tối đa của người lao động là bao lâu?"):
            sample_q = "thời gian thử việc của người lao động tối đa bao lâu"
        if col3.button("💰 Tiền lương làm thêm giờ vào ngày nghỉ lễ tính thế nào?"):
            sample_q = "Tiền lương làm thêm giờ vào ngày nghỉ lễ được tính thế nào"

        # Hiển thị các tin nhắn cũ
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "structured" in msg and msg["structured"]:
                    grounding = msg["structured"].get("grounding", {})
                    badge = grounding.get("badge", "")
                    summary = grounding.get("summary", "")
                    if badge:
                        if "VERIFIED" in grounding.get("verdict", ""):
                            st.success(f"**{badge}**: {summary}")
                        elif "HALLUCINATION_RISK" in grounding.get("verdict", ""):
                            st.error(f"**{badge}**: {summary}")
                        elif "ABSTAINED" in grounding.get("verdict", ""):
                            st.info(f"**{badge}**: {summary}")
                        else:
                            st.warning(f"**{badge}**: {summary}")

                    with st.expander("🛡️ Chi Tiết Kiểm Chứng Căn Cứ & Trích Dẫn (Grounding - Phase 15)"):
                        st.write(f"- **Điểm tin cậy trích dẫn (Fidelity):** `{grounding.get('grounding_score', 0.0) * 100:.0f}%`")
                        st.write(f"- **Phán quyết:** `{grounding.get('verdict', 'N/A')}`")
                        v_items = grounding.get("verified_laws", []) + grounding.get("verified_articles", [])
                        u_items = grounding.get("unverified_laws", []) + grounding.get("unverified_articles", [])
                        if v_items:
                            st.markdown(f"✅ **Căn cứ đã xác thực khớp tài liệu:** {', '.join(v_items)}")
                        if u_items:
                            st.markdown(f"❌ **Căn cứ không tìm thấy trong tài liệu:** {', '.join(u_items)}")

                    with st.expander("🧩 Dữ Liệu Có Cấu Trúc (Structured JSON API)"):
                        st.json(msg["structured"])

                if "docs" in msg and msg["docs"]:
                    with st.expander(f"📚 Xem {len(msg['docs'])} Căn Cứ Pháp Lý Được Trích Xuất"):
                        for doc in msg["docs"]:
                            law_label = doc.get("law_id") or "N/A"
                            aid_label = doc.get("aid") or str(doc.get("article_id", "N/A"))
                            st.markdown(
                                f'<span class="law-badge">Văn bản: {law_label}</span>'
                                f'<span class="law-badge">Điều: {aid_label}</span>'
                                f'<span class="score-badge">Điểm: {doc.get("score", 0.0):.4f}</span>',
                                unsafe_allow_html=True,
                            )
                            if doc.get("is_compressed"):
                                st.caption(f"✂️ *Đã nén {int((1.0 - doc.get('compression_ratio', 1.0)) * 100)}% dung lượng — Giữ lại: {', '.join(doc.get('clauses_retained', []))}*")
                            st.markdown(f"```\n{doc.get('text', '')}\n```")
                            st.divider()

        user_input = st.chat_input("Nhập câu hỏi pháp lý của bạn...")
        prompt_to_run = sample_q or user_input

        if prompt_to_run:
            # Multi-turn contextualization
            contextualized_query, is_follow_up, prior_topic = contextualizer.contextualize(
                prompt_to_run, st.session_state.messages
            )
            st.session_state.messages.append({"role": "user", "content": prompt_to_run})
            with st.chat_message("user"):
                st.markdown(prompt_to_run)

            # Acronym normalization
            search_query, detected_acronyms, _ = normalizer.normalize(contextualized_query)

            # Retrieval
            with st.spinner("⚡ Đang quét chỉ mục và truy xuất điều luật..."):
                t0 = time.time()
                if "Reranker" in retriever_choice:
                    retrieved_docs = search_hybrid_rerank(
                        search_query, top_k, bm25_retriever, dense_retriever, reranker, corpus
                    )
                elif "Hybrid" in retriever_choice:
                    retrieved_docs = search_hybrid(search_query, top_k, bm25_retriever, dense_retriever, corpus)
                elif "Dense" in retriever_choice:
                    retrieved_docs = search_dense(search_query, top_k, dense_retriever, corpus)
                else:
                    retrieved_docs = search_bm25(search_query, top_k, bm25_retriever, corpus)
                retrieval_ms = (time.time() - t0) * 1000

            # Answer synthesis
            with st.chat_message("assistant"):
                status_text = f"⚡ Tìm thấy {len(retrieved_docs)} điều luật trong {retrieval_ms:.1f}ms"
                if is_follow_up and prior_topic:
                    status_text += f" | 🔄 Tiếp nối chủ đề: *'{prior_topic}'*"
                if detected_acronyms:
                    exp_list = ", ".join([f"**{k}** ➔ *{v}*" for k, v in detected_acronyms.items()])
                    status_text += f" | 🔍 Chuẩn hóa viết tắt: {exp_list}"
                user_prompt, system_prompt, compressed_docs = build_prompts(
                    contextualized_query, retrieved_docs, compressor=compressor
                )
                st.caption(status_text)

                full_response = ""
                t_llm0 = time.time()
                if "Colab" in backend_choice:
                    if not colab_url:
                        full_response = "⚠️ **Vui lòng dán Colab Tunnel URL** vào thanh cài đặt bên trái."
                        st.warning(full_response)
                    else:
                        with st.spinner("🧠 GPU Colab đang suy luận câu trả lời..."):
                            response_gen = stream_colab(user_prompt, system_prompt, colab_url)
                            full_response = st.write_stream(response_gen)
                elif "Groq" in backend_choice:
                    if not groq_api_key:
                        full_response = "⚠️ **Vui lòng nhập Groq API Key** ở thanh cài đặt bên trái."
                        st.warning(full_response)
                    else:
                        response_gen = stream_groq(user_prompt, system_prompt, groq_api_key, groq_model)
                        full_response = st.write_stream(response_gen)
                elif "Gemini" in backend_choice:
                    if not gemini_api_key:
                        full_response = "⚠️ **Vui lòng nhập Gemini API Key** ở thanh cài đặt bên trái."
                        st.warning(full_response)
                    else:
                        response_gen = stream_gemini(user_prompt, system_prompt, gemini_api_key, gemini_model)
                        full_response = st.write_stream(response_gen)
                elif "Ollama" in backend_choice:
                    response_gen = stream_ollama(user_prompt, system_prompt, ollama_model)
                    full_response = st.write_stream(response_gen)
                else:
                    full_response = (
                        "**[Chế độ Tra cứu - Không LLM]**\n\n"
                        "Dưới đây là các điều luật có độ khớp cao nhất được hệ thống tìm thấy dựa trên câu hỏi của bạn."
                    )
                    st.markdown(full_response)
                llm_ms = (time.time() - t_llm0) * 1000

                # Grounding & structured data
                structured_data = None
                grounding = {}
                if "Không LLM" not in full_response and full_response.strip():
                    structured_data = LegalResponseFormatter.to_api_json(full_response, compressed_docs)
                    grounding = structured_data.get("grounding", {})
                    badge = grounding.get("badge", "")
                    summary = grounding.get("summary", "")

                    if badge:
                        if "VERIFIED" in grounding.get("verdict", ""):
                            st.success(f"**{badge}**: {summary}")
                        elif "HALLUCINATION_RISK" in grounding.get("verdict", ""):
                            st.error(f"**{badge}**: {summary}")
                        elif "ABSTAINED" in grounding.get("verdict", ""):
                            st.info(f"**{badge}**: {summary}")
                        else:
                            st.warning(f"**{badge}**: {summary}")

                    with st.expander("🛡️ Chi Tiết Kiểm Chứng Căn Cứ & Trích Dẫn (Grounding - Phase 15)"):
                        st.write(f"- **Điểm tin cậy trích dẫn (Fidelity):** `{grounding.get('grounding_score', 0.0) * 100:.0f}%`")
                        st.write(f"- **Phán quyết:** `{grounding.get('verdict', 'N/A')}`")
                        v_items = grounding.get("verified_laws", []) + grounding.get("verified_articles", [])
                        u_items = grounding.get("unverified_laws", []) + grounding.get("unverified_articles", [])
                        if v_items:
                            st.markdown(f"✅ **Căn cứ đã xác thực khớp tài liệu:** {', '.join(v_items)}")
                        if u_items:
                            st.markdown(f"❌ **Căn cứ không tìm thấy trong tài liệu:** {', '.join(u_items)}")

                    with st.expander("🧩 Dữ Liệu Có Cấu Trúc (Structured JSON API)"):
                        st.json(structured_data)

                if compressed_docs:
                    with st.expander(f"📚 Xem {len(compressed_docs)} Căn Cứ Pháp Lý Được Trích Xuất"):
                        for doc in compressed_docs:
                            law_label = doc.get("law_id") or "N/A"
                            aid_label = doc.get("aid") or str(doc.get("article_id", "N/A"))
                            st.markdown(
                                f'<span class="law-badge">Văn bản: {law_label}</span>'
                                f'<span class="law-badge">Điều: {aid_label}</span>'
                                f'<span class="score-badge">Điểm: {doc.get("score", 0.0):.4f}</span>',
                                unsafe_allow_html=True,
                            )
                            if doc.get("is_compressed"):
                                st.caption(f"✂️ *Đã nén {int((1.0 - doc.get('compression_ratio', 1.0)) * 100)}% dung lượng — Giữ lại: {', '.join(doc.get('clauses_retained', []))}*")
                            st.markdown(f"```\n{doc.get('text', '')}\n```")
                            if doc.get("raw_text") and doc.get("is_compressed"):
                                with st.expander("📄 Xem toàn văn điều luật gốc (chưa nén)"):
                                    st.text(doc.get("raw_text", ""))
                            st.divider()

            # Lưu vào session
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "docs": compressed_docs,
                    "structured": structured_data,
                }
            )
            # Log analytics
            st.session_state.query_logs.append(
                {
                    "Thời gian": time.strftime("%H:%M:%S"),
                    "Câu hỏi": prompt_to_run,
                    "Phương pháp": retriever_choice.split()[0],
                    "Mô hình": backend_choice.split()[0],
                    "Độ trễ tìm kiếm (ms)": round(retrieval_ms, 1),
                    "Độ trễ LLM (ms)": round(llm_ms, 1),
                    "Điểm tin cậy": f"{grounding.get('grounding_score', 0.0) * 100:.0f}%" if structured_data else "N/A",
                    "Phán quyết": grounding.get("verdict", "N/A") if structured_data else "N/A",
                }
            )

    # =========================================================================
    # TAB 2: TRA CỨU ĐIỀU LUẬT (FAST SEARCH)
    # =========================================================================
    with tab_search:
        st.subheader("🔍 Tìm Kiếm Nhanh Điều Luật (Retrieval-Only)")
        st.caption("Tìm kiếm siêu tốc dựa trên Inverted Index (BM25) và Semantic Dense Index (BGE-M3), không cần bật LLM.")

        c_query, c_method, c_k = st.columns([3, 1.5, 1])
        search_kw = c_query.text_input("Từ khóa hoặc nội dung điều luật:", placeholder="Ví dụ: thời gian thử việc, nghỉ thai sản, hợp đồng mùa vụ...")
        search_method = c_method.selectbox(
            "Phương pháp:",
            ["Hybrid + Reranker", "Hybrid", "Dense (BGE-M3)", "BM25"],
            index=0,
            key="fast_search_method",
        )
        search_topk = c_k.number_input("Top-K:", min_value=1, max_value=20, value=5, step=1)

        if st.button("🚀 Bắt Đầu Tìm Kiếm", type="primary", use_container_width=True) or search_kw:
            if not search_kw.strip():
                st.info("Vui lòng nhập từ khóa để tìm kiếm.")
            else:
                norm_kw, acronyms, _ = normalizer.normalize(search_kw)
                if acronyms:
                    st.info(f"💡 Đã nhận diện và chuẩn hóa từ viết tắt: {', '.join([f'{k} -> {v}' for k, v in acronyms.items()])}")

                t_s0 = time.time()
                with st.spinner("Đang tìm kiếm..."):
                    if "Reranker" in search_method:
                        results = search_hybrid_rerank(norm_kw, search_topk, bm25_retriever, dense_retriever, reranker, corpus)
                    elif "Hybrid" in search_method:
                        results = search_hybrid(norm_kw, search_topk, bm25_retriever, dense_retriever, corpus)
                    elif "Dense" in search_method:
                        results = search_dense(norm_kw, search_topk, dense_retriever, corpus)
                    else:
                        results = search_bm25(norm_kw, search_topk, bm25_retriever, corpus)
                search_time = (time.time() - t_s0) * 1000

                st.success(f"Tìm thấy **{len(results)} điều luật** phù hợp trong **{search_time:.1f}ms**:")

                for r in results:
                    with st.container():
                        law_label = r.get("law_id") or "N/A"
                        aid_label = r.get("aid") or str(r.get("article_id", "N/A"))
                        st.markdown(
                            f'<div class="result-card">'
                            f'<b>Top {r["rank"]}</b> | '
                            f'<span class="law-badge">Văn bản: {law_label}</span>'
                            f'<span class="law-badge">Điều: {aid_label}</span>'
                            f'<span class="score-badge">Độ liên quan: {r["score"]:.4f}</span>'
                            f'<br><br><b>Nội dung trích đoạn:</b>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        st.code(r["text"], language="text")

    # =========================================================================
    # TAB 3: THƯ VIỆN VĂN BẢN (DOCUMENT EXPLORER)
    # =========================================================================
    with tab_docs:
        st.subheader("📚 Thư Viện Văn Bản Pháp Luật (Document Explorer)")
        st.caption(f"Duyệt và đọc toàn văn trong kho lưu trữ gồm **{len(corpus):,} điều luật** thuộc **{len(laws_catalog):,} văn bản quy phạm pháp luật**.")

        doc_mode = st.radio(
            "Chế độ xem:",
            ["📖 Duyệt theo Văn bản Luật & Bộ luật", "🎯 Tra cứu trực tiếp theo Global Article ID"],
            horizontal=True,
        )

        if "Global Article ID" in doc_mode:
            aid_input = st.number_input("Nhập ID Điều luật (từ 0 đến 59,635):", min_value=0, max_value=len(corpus) - 1, value=54231)
            doc_item = corpus.get(int(aid_input))
            if doc_item:
                st.success(f"Tìm thấy Điều luật **#{aid_input}** thuộc văn bản **{doc_item['law_id']}**")
                st.markdown(f"**Số hiệu Điều:** `{doc_item['aid']}`")
                st.text_area("Toàn văn điều luật:", value=doc_item["text"], height=350)
            else:
                st.error(f"Không tìm thấy điều luật với ID: {aid_input}")
        else:
            # Major law quick buttons
            st.markdown("**Các Bộ luật trọng tâm thường dùng:**")
            quick_cols = st.columns(5)
            selected_law = None
            if quick_cols[0].button("Bộ luật Lao động 2019"):
                selected_law = "45/2019/QH14"
            if quick_cols[1].button("Luật Doanh nghiệp 2020"):
                selected_law = "59/2020/QH14"
            if quick_cols[2].button("Bộ luật Dân sự 2015"):
                selected_law = "91/2015/QH13"
            if quick_cols[3].button("Bộ luật Hình sự 2015"):
                selected_law = "100/2015/QH13"
            if quick_cols[4].button("Nghị định 145/2020"):
                selected_law = "145/2020/NĐ-CP"

            all_laws = sorted(list(laws_catalog.keys()))
            default_law_idx = 0
            if selected_law and selected_law in all_laws:
                default_law_idx = all_laws.index(selected_law)

            chosen_law = st.selectbox("Chọn mã hiệu văn bản pháp luật:", all_laws, index=default_law_idx)

            article_ids_in_law = laws_catalog.get(chosen_law, [])
            st.info(f"Văn bản **{chosen_law}** có tổng cộng **{len(article_ids_in_law)} điều luật** được lập chỉ mục.")

            # Dropdown điều luật thuộc văn bản
            article_options = []
            for a_id in article_ids_in_law:
                doc_obj = corpus.get(a_id, {})
                first_line = doc_obj.get("text", "").split("\n")[0][:80]
                article_options.append(f"Điều ID {doc_obj.get('aid', 'N/A')} (Global #{a_id}) - {first_line}...")

            chosen_article_str = st.selectbox("Chọn điều luật để đọc:", article_options)
            if chosen_article_str:
                # Extract Global ID
                import re

                match = re.search(r"Global #(\d+)", chosen_article_str)
                if match:
                    chosen_aid = int(match.group(1))
                    article_data = corpus.get(chosen_aid, {})
                    st.markdown(f"### 📄 Toàn Văn: {chosen_law} — {article_data.get('aid', '')}")
                    st.code(article_data.get("text", ""), language="text")

    # =========================================================================
    # TAB 4: LỊCH SỬ & THỐNG KÊ (ANALYTICS)
    # =========================================================================
    with tab_history:
        st.subheader("📊 Nhật Ký & Thống Kê Truy Vấn (Query Analytics)")
        st.caption("Theo dõi hiệu năng hệ thống, độ trễ từng bước và chất lượng trích dẫn trong phiên làm việc.")

        total_queries = len(st.session_state.query_logs)

        # 4 Metric cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tổng số câu hỏi", f"{total_queries}")
        if total_queries > 0:
            avg_search = sum(l["Độ trễ tìm kiếm (ms)"] for l in st.session_state.query_logs) / total_queries
            avg_llm = sum(l["Độ trễ LLM (ms)"] for l in st.session_state.query_logs) / total_queries
            m2.metric("Độ trễ tìm kiếm TB", f"{avg_search:.1f} ms")
            m3.metric("Độ trễ LLM TB", f"{avg_llm:.1f} ms")
            m4.metric("Trạng thái", "Sẵn sàng ✅")
        else:
            m2.metric("Độ trễ tìm kiếm TB", "0 ms")
            m3.metric("Độ trễ LLM TB", "0 ms")
            m4.metric("Trạng thái", "Chờ câu hỏi")

        st.divider()

        if total_queries > 0:
            st.markdown("#### Danh Sách Các Lượt Truy Vấn:")
            log_df = pd.DataFrame(st.session_state.query_logs)
            st.dataframe(log_df, use_container_width=True)

            st.download_button(
                label="📥 Tải xuống lịch sử hội thoại (JSON)",
                data=json.dumps(st.session_state.messages, ensure_ascii=False, indent=2),
                file_name="legal_rag_history.json",
                mime="application/json",
            )
        else:
            st.info("Chưa có câu hỏi nào được thực hiện trong phiên làm việc này. Hãy quay lại Tab 1 để đặt câu hỏi!")


if __name__ == "__main__":
    main()
