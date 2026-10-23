"""Core Service layer for Vietnamese Legal RAG REST API."""

import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import pyarrow.parquet as pq
import requests

from src.rag.citation.grounding import GroundingVerifier
from src.rag.context.compressor import LegalPassageCompressor
from src.rag.output.formatter import LegalResponseFormatter
from src.rag.query.contextualizer import ConversationContextualizer
from src.rag.query.normalizer import LegalAcronymNormalizer
from src.retrieval.bm25_retriever import BM25Retriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_retrieval"
    / "corpus_articles_preprocessed.parquet"
)
DEFAULT_BM25_INDEX_PATH = PROJECT_ROOT / "data" / "indices" / "bm25_index.pkl"
DEFAULT_DENSE_INDEX_DIR = PROJECT_ROOT / "data" / "indices" / "dense_bge_m3"


class LegalRAGService:
    """Production service managing RAG models, indices, search, and synthesis."""

    def __init__(
        self,
        corpus_path: Optional[Path] = None,
        bm25_index_path: Optional[Path] = None,
        dense_index_dir: Optional[Path] = None,
        lazy_load: bool = True,
    ):
        self.corpus_path = corpus_path or DEFAULT_CORPUS_PATH
        self.bm25_index_path = bm25_index_path or DEFAULT_BM25_INDEX_PATH
        self.dense_index_dir = dense_index_dir or DEFAULT_DENSE_INDEX_DIR
        self.start_time = time.time()

        self.corpus: Dict[int, Dict[str, Any]] = {}
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.dense_retriever: Any = None
        self.encoder: Any = None
        self.reranker: Any = None

        self.normalizer = LegalAcronymNormalizer()
        self.contextualizer = ConversationContextualizer()
        self.compressor = LegalPassageCompressor()
        self.grounding_verifier = GroundingVerifier()

        self._is_loaded = False
        if not lazy_load:
            self.load()

    def load(
        self,
        load_dense: bool = True,
        load_reranker: bool = True,
    ) -> None:
        """Load corpus, BM25, Dense FAISS index, and Reranker model."""
        if self._is_loaded:
            return

        # 1. Load BM25 index
        if self.bm25_index_path.exists():
            self.bm25_retriever = BM25Retriever.load(self.bm25_index_path)

        # 2. Load Dense BGE-M3 retriever
        if load_dense and self.dense_index_dir.exists():
            from src.retrieval.dense.encoder import DenseEncoder
            from src.retrieval.dense.retriever import DenseRetriever

            self.encoder = DenseEncoder()
            self.dense_retriever = DenseRetriever.load(
                self.dense_index_dir, encoder=self.encoder
            )

        # 3. Load Cross-Encoder Reranker
        if load_reranker:
            from src.retrieval.reranking.cross_encoder import ArticleReranker

            self.reranker = ArticleReranker(device="cpu", max_length=384)

        # 4. Load Legal Corpus
        if self.corpus_path.exists():
            table = pq.read_table(
                self.corpus_path,
                columns=["global_article_id", "law_id", "aid", "text"],
            )
            df = table.to_pandas()
            for _, row in df.iterrows():
                self.corpus[int(row["global_article_id"])] = {
                    "law_id": str(row["law_id"]),
                    "aid": str(row["aid"]),
                    "text": str(row["text"]),
                }

        self._is_loaded = True

    def get_document(self, article_id: int) -> Optional[Dict[str, Any]]:
        """Lookup an article by its global integer ID."""
        if not self._is_loaded:
            self.load()
        doc = self.corpus.get(article_id)
        if not doc:
            return None
        return {
            "article_id": article_id,
            "law_id": doc.get("law_id", ""),
            "aid": doc.get("aid", ""),
            "text": doc.get("text", ""),
        }

    def _search_hybrid(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        bm25_res = (
            self.bm25_retriever.search(query, top_k=top_k * 2)
            if self.bm25_retriever
            else []
        )
        dense_res = (
            self.dense_retriever.search(query, top_k=top_k * 2)
            if self.dense_retriever
            else []
        )

        rrf_scores: Dict[int, float] = {}
        for rank, r in enumerate(bm25_res, start=1):
            rrf_scores[r.article_id] = rrf_scores.get(r.article_id, 0.0) + (
                1.0 / (60 + rank)
            )
        for rank, r in enumerate(dense_res, start=1):
            rrf_scores[r.article_id] = rrf_scores.get(r.article_id, 0.0) + (
                1.0 / (60 + rank)
            )

        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )[:top_k]
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

    def search(
        self,
        query: str,
        retriever_type: str = "hybrid_rerank",
        top_k: int = 5,
    ) -> Tuple[List[Dict[str, Any]], str, float]:
        """Search legal articles with specified retriever. Returns (docs, norm_query, latency_ms)."""
        if not self._is_loaded:
            self.load()
        t0 = time.time()
        norm_query, _, _ = self.normalizer.normalize(query)
        retriever_choice = retriever_type.lower()

        if retriever_choice == "hybrid_rerank":
            pool_size = max(15, top_k * 2)
            candidates = self._search_hybrid(norm_query, top_k=pool_size)
            if not self.reranker or not candidates:
                docs = candidates[:top_k]
            else:
                candidate_ids = [c["article_id"] for c in candidates]
                candidate_texts = [c["text"] for c in candidates]
                rerank_results = self.reranker.rerank(
                    query=norm_query,
                    candidate_ids=candidate_ids,
                    candidate_texts=candidate_texts,
                    top_k=top_k,
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
        elif retriever_choice == "hybrid":
            docs = self._search_hybrid(norm_query, top_k=top_k)
        elif retriever_choice == "dense":
            results = (
                self.dense_retriever.search(norm_query, top_k=top_k)
                if self.dense_retriever
                else []
            )
            docs = [
                {
                    "rank": r.rank,
                    "article_id": r.article_id,
                    "score": float(r.score),
                    "law_id": self.corpus.get(r.article_id, {}).get("law_id", ""),
                    "aid": self.corpus.get(r.article_id, {}).get("aid", ""),
                    "text": self.corpus.get(r.article_id, {}).get("text", ""),
                }
                for r in results
            ]
        else:  # bm25
            results = (
                self.bm25_retriever.search(norm_query, top_k=top_k)
                if self.bm25_retriever
                else []
            )
            docs = [
                {
                    "rank": r.rank,
                    "article_id": r.article_id,
                    "score": float(r.score),
                    "law_id": self.corpus.get(r.article_id, {}).get("law_id", ""),
                    "aid": self.corpus.get(r.article_id, {}).get("aid", ""),
                    "text": self.corpus.get(r.article_id, {}).get("text", ""),
                }
                for r in results
            ]

        latency_ms = (time.time() - t0) * 1000
        return docs, norm_query, latency_ms

    def build_prompt(
        self, query: str, docs: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """Build structured system prompt and context-injected user prompt."""
        system_prompt = LegalResponseFormatter.SYSTEM_PROMPT_TEMPLATE

        top_docs = self.compressor.compress_documents(query, docs[:3])
        context_blocks = []
        for d in top_docs:
            snippet = d["text"].strip()
            law_meta = (
                f"Văn bản ID: {d['law_id']} | Điều ID: {d['aid']}"
                if d.get("law_id")
                else f"Article #{d.get('article_id')}"
            )
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

    def _call_llm(
        self,
        backend: str,
        user_prompt: str,
        system_prompt: str,
        colab_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Call external or local LLM server."""
        if backend == "colab":
            endpoint = colab_url or os.environ.get("COLAB_URL")
            if not endpoint:
                raise ValueError("colab_url is required when backend='colab'")
            url = endpoint.strip().rstrip("/")
            if not url.endswith("/v1/chat/completions"):
                url = f"{url}/v1/chat/completions"
            payload = {
                "model": model or "qwen2.5-7b",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            resp = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

        elif backend == "ollama":
            url = "http://localhost:11434/api/chat"
            payload = {
                "model": model or "qwen2.5:1.5b",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": temperature},
            }
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "")

        elif backend in ("groq", "openai"):
            key = (
                api_key
                or (
                    os.environ.get("GROQ_API_KEY")
                    if backend == "groq"
                    else os.environ.get("OPENAI_API_KEY")
                )
            )
            if not key:
                raise ValueError(f"API key is required for backend='{backend}'")
            api_base = (
                "https://api.groq.com/openai/v1/chat/completions"
                if backend == "groq"
                else "https://api.openai.com/v1/chat/completions"
            )
            payload = {
                "model": model
                or (
                    "llama-3.3-70b-versatile"
                    if backend == "groq"
                    else "gpt-4o-mini"
                ),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            resp = requests.post(
                api_base,
                json=payload,
                headers={
                    "Authorization": f"Bearer {key.strip()}",
                    "Content-Type": "application/json",
                },
                timeout=60,
            )
            resp.raise_for_status()
            return (
                resp.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

        elif backend == "gemini":
            key = api_key or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise ValueError("GEMINI_API_KEY is required for backend='gemini'")
            m_name = model or "gemini-2.0-flash"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={key.strip()}"
            full_prompt = f"{system_prompt}\n\n---\n{user_prompt}"
            payload = {
                "contents": [{"parts": [{"text": full_prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            }
            resp = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            resp.raise_for_status()
            return (
                resp.json()
                .get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )

        return ""

    def answer_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        backend: str = "auto",
        colab_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        retriever_type: str = "hybrid_rerank",
        top_k: int = 5,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Full RAG execution: Multi-turn -> Normalizer -> Search -> Compress -> LLM -> Grounding."""
        t_total_start = time.time()

        # 1. Multi-turn contextualization
        history = conversation_history or []
        ctx_query, is_follow_up, _ = self.contextualizer.contextualize(
            query, history
        )

        # 2. Acronym normalization
        norm_query, detected_acronyms, _ = self.normalizer.normalize(ctx_query)

        # 3. Retrieval
        docs, _, search_time = self.search(
            query=norm_query, retriever_type=retriever_type, top_k=top_k
        )

        # 4. Prompt construction with passage compression
        user_prompt, system_prompt = self.build_prompt(ctx_query, docs)

        # 5. Resolve backend
        effective_backend = backend
        if effective_backend == "auto":
            if colab_url or os.environ.get("COLAB_URL"):
                effective_backend = "colab"
            elif api_key or os.environ.get("GROQ_API_KEY"):
                effective_backend = "groq"
            elif os.environ.get("GEMINI_API_KEY"):
                effective_backend = "gemini"
            elif os.environ.get("OPENAI_API_KEY"):
                effective_backend = "openai"
            else:
                effective_backend = "none"

        # 6. LLM Generation
        t_gen_start = time.time()
        raw_answer = ""
        if effective_backend != "none":
            try:
                raw_answer = self._call_llm(
                    backend=effective_backend,
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    colab_url=colab_url,
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as err:
                raw_answer = f"[Lỗi kết nối Backend {effective_backend}]: {str(err)}"
        gen_time = (time.time() - t_gen_start) * 1000

        # 7. Response formatting and parsing
        parsed_sections = LegalResponseFormatter.parse_sections(raw_answer)

        # 8. Grounding verification
        grounding_result = None
        if raw_answer and effective_backend != "none" and not parsed_sections["is_abstention"]:
            ver_res = self.grounding_verifier.verify(raw_answer, docs)
            grounding_result = {
                "score": ver_res.score,
                "verdict": ver_res.verdict,
                "badge": ver_res.badge,
                "summary": ver_res.summary,
                "cited_laws": ver_res.cited_laws,
                "cited_articles": ver_res.cited_articles,
                "missing_citations": ver_res.missing_citations,
                "ungrounded_citations": ver_res.ungrounded_citations,
            }

        total_time = (time.time() - t_total_start) * 1000

        return {
            "status": "SUCCESS",
            "query": query,
            "rewritten_query": ctx_query if is_follow_up else None,
            "detected_acronyms": detected_acronyms or None,
            "answer": raw_answer or "(Không có câu trả lời từ LLM)",
            "direct_conclusion": parsed_sections["direct_conclusion"],
            "legal_grounds": parsed_sections["legal_grounds"],
            "notes_and_conditions": parsed_sections["notes_and_conditions"],
            "is_abstention": parsed_sections["is_abstention"],
            "grounding": grounding_result,
            "sources": docs,
            "latency": {
                "search_ms": round(search_time, 2),
                "generation_ms": round(gen_time, 2),
                "total_ms": round(total_time, 2),
            },
        }
