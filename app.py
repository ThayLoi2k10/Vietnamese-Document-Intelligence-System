"""Vietnamese Legal RAG Assistant - Streamlit Web Application (Phase 18 UI)."""

try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

def _load_env_file() -> None:
    """Nạp biến môi trường từ .env (không phụ thuộc python-dotenv)."""
    env_path = __import__("pathlib").Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    import os as _os
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip("'\"")
        if key and val and not _os.environ.get(key):
            _os.environ[key] = val


_load_env_file()


import json
import os
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
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
DEFAULT_HYBRID_CONFIG_PATH = PROJECT_ROOT / "data" / "indices" / "hybrid_config.json"

# BM25 tuned chính thức (Phase 5): k1=1.80, b=0.85
BM25_TUNED_K1 = 1.80
BM25_TUNED_B = 0.85

# Hai phương pháp truy xuất hiển thị cho người dùng
RETRIEVER_HYBRID = "Hybrid (BM25 + BGE-M3) — chính xác nhất"
RETRIEVER_BM25_LORA = "BM25 + LoRA Reranker (Fine-tuned) — chính xác nhất"
RETRIEVER_BM25 = "BM25 — nhanh, nhẹ"

# Thứ tự ưu tiên model Groq (model thực tế được lấy từ danh sách live của Groq)
GROQ_MODEL_PRIORITY = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def get_secret_or_env(key: str, default: str = "") -> str:
    """Safe helper to fetch secret from os.environ or st.secrets without crashing."""
    val = os.environ.get(key, "")
    if val:
        return val
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


HF_BASE_URL = "https://huggingface.co/spaces/ThayLoi2k10/vietnamese-legal-rag/resolve/main"

def _ensure_remote_file(path: Path, rel_url: str, min_size: int = 1_000_000):
    if path.exists() and path.stat().st_size >= min_size:
        return
    if path.exists() and path.stat().st_size == 0:
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
    import gc

    _ensure_remote_file(DEFAULT_CORPUS_PATH, "data/processed/legal_retrieval/corpus_articles_preprocessed.parquet", min_size=10_000_000)
    corpus: Dict[int, Dict[str, Any]] = {}
    if DEFAULT_CORPUS_PATH.exists():
        try:
            table = pq.read_table(
                DEFAULT_CORPUS_PATH,
                columns=["global_article_id", "law_id", "aid", "text"],
            )
            gids = table["global_article_id"].to_pylist()
            lids = table["law_id"].to_pylist()
            aids = table["aid"].to_pylist()
            txts = table["text"].to_pylist()
            del table
            corpus = {
                int(g): {"law_id": str(l), "aid": str(a), "text": str(x)}
                for g, l, a, x in zip(gids, lids, aids, txts)
            }
            del gids, lids, aids, txts
            gc.collect()
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
        retriever = BM25Retriever.load(DEFAULT_BM25_INDEX_PATH)
        # Dùng cấu hình BM25 tuned chính thức (không cần rebuild postings)
        retriever.k1 = BM25_TUNED_K1
        retriever.b = BM25_TUNED_B
        return retriever
    return None


@st.cache_resource(show_spinner="Đang nạp mô hình ngữ nghĩa BGE-M3 + chỉ mục vector (lần đầu ~30-60 giây)...")
def load_hybrid(_bm25_retriever):
    """Nạp Hybrid Weighted RRF (Phase 11) = BGE-M3 dense + BM25 tuned.

    Trả về (retriever, error_message). Nếu lỗi, retriever=None để app tự dùng BM25.
    """
    try:
        if _bm25_retriever is None:
            return None, "Chưa có chỉ mục BM25."
        if not (DEFAULT_DENSE_INDEX_DIR / "embeddings.f32.npy").exists():
            _ensure_remote_file(
                DEFAULT_DENSE_INDEX_DIR / "index.faiss",
                "data/indices/dense_bge_m3/index.faiss",
                min_size=200_000_000,
            )
            _ensure_remote_file(
                DEFAULT_DENSE_INDEX_DIR / "corpus_ids.json",
                "data/indices/dense_bge_m3/corpus_ids.json",
                min_size=100_000,
            )
        has_dense_index = (
            (DEFAULT_DENSE_INDEX_DIR / "embeddings.f32.npy").exists()
            or (DEFAULT_DENSE_INDEX_DIR / "index.faiss").exists()
        )
        if not has_dense_index:
            return None, "Chưa có chỉ mục vector BGE-M3 trên môi trường này."
        from src.retrieval.dense_hf_retriever import HFDenseRetriever
        from src.retrieval.hybrid.retriever import HybridRetriever

        dense = HFDenseRetriever.load(DEFAULT_DENSE_INDEX_DIR)
        dense.search("khởi động mô hình", top_k=1)  # warm-up: tải model ngay lúc khởi động
        if DEFAULT_HYBRID_CONFIG_PATH.exists():
            hybrid = HybridRetriever.load(
                DEFAULT_HYBRID_CONFIG_PATH, dense_retriever=dense, bm25_retriever=_bm25_retriever
            )
        else:
            hybrid = HybridRetriever(dense, _bm25_retriever, "weighted_rrf", 5, 0.75, 0.25, 100)
        return hybrid, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_resource(show_spinner="Đang nạp LoRA Legal Reranker đã huấn luyện...")
def load_finetuned_reranker():
    weights_path = PROJECT_ROOT / "data" / "indices" / "reranker_finetuned" / "lora_reranker_weights.pt"
    if not weights_path.exists():
        return None
    try:
        from src.retrieval.reranking.finetuned_reranker import FinetunedLegalReranker

        reranker = FinetunedLegalReranker(weights_path=weights_path)
        reranker._init_model()
        return reranker if reranker.is_finetuned else None
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


def run_search(
    query: str, top_k: int, retriever, corpus: Dict[int, Dict[str, Any]], reranker=None
) -> List[Dict[str, Any]]:
    if retriever is None:
        return []
    if reranker is not None:
        pool_k = max(top_k, 10)  # giới hạn Top-10 khi chạy tương tác trên CPU để giảm một nửa độ trễ
        first_stage = retriever.search(query, top_k=pool_k)
        cand_ids = [r.article_id for r in first_stage]
        cand_priors = [float(r.score) for r in first_stage]
        cand_texts = []
        for aid in cand_ids:
            meta = corpus.get(aid, {})
            lid, a_str, txt = meta.get("law_id", ""), meta.get("aid", ""), meta.get("text", "")
            header = f"[Văn bản: {lid} | Điều: {a_str}] " if lid else ""
            full_t = (header + txt).strip()
            if len(full_t) > 1600:
                full_t = full_t[:1200].rstrip() + "\n...\n" + full_t[-400:].lstrip()
            cand_texts.append(full_t)
        results = reranker.rerank(
            query=query,
            candidate_ids=cand_ids,
            candidate_texts=cand_texts,
            candidate_prior_scores=cand_priors,
            candidate_pool_size=pool_k,
            top_k=top_k,
        )
    else:
        results = retriever.search(query, top_k=top_k)

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


def build_prompts(
    query: str, docs: List[Dict[str, Any]], compressor: Optional[LegalPassageCompressor] = None
) -> tuple[str, str, List[Dict[str, Any]]]:
    system_prompt = LegalResponseFormatter.SYSTEM_PROMPT_TEMPLATE

    active_docs = docs[:5]
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


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_groq_model_ids(api_key: str) -> List[str]:
    """Lấy danh sách model Groq. Ném exception khi lỗi để st.cache_data KHÔNG cache lỗi."""
    last_exc: Exception = RuntimeError("unknown")
    for _ in range(3):
        try:
            resp = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=20,
            )
        except requests.RequestException as exc:
            last_exc = exc
            continue
        if resp.status_code in (401, 403):
            raise PermissionError(f"Groq từ chối key (HTTP {resp.status_code}).")
        if resp.status_code == 200:
            return [m.get("id", "") for m in resp.json().get("data", [])]
        last_exc = RuntimeError(f"HTTP {resp.status_code}")
    raise last_exc


def pick_groq_model(api_key: str) -> tuple[str, str]:
    """Kiểm tra Groq key + chọn model chat còn hoạt động. Trả về (model, error)."""
    if not api_key:
        return "", "Chưa cấu hình GROQ_API_KEY."
    try:
        ids = _fetch_groq_model_ids(api_key)
    except PermissionError as exc:
        return "", str(exc)
    except Exception:  # noqa: BLE001
        # Mạng chậm/timeout: vẫn dùng model ưu tiên; nếu lỗi thật sẽ tự lùi về Ollama khi trả lời
        return GROQ_MODEL_PRIORITY[0], ""
    for m in GROQ_MODEL_PRIORITY:
        if m in ids:
            return m, ""
    excluded = ["whisper", "guard", "tts", "orpheus", "embed", "safeguard", "allam"]
    chat = [m for m in ids if m and not any(k in m.lower() for k in excluded)]
    if chat:
        return chat[0], ""
    return "", "Groq không còn model chat khả dụng."


@st.cache_data(ttl=120, show_spinner=False)
def pick_ollama_model() -> str:
    """Trả về tên model Ollama local đầu tiên (nếu Ollama đang chạy), ngược lại ''."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            models = [m for m in models if m and "embed" not in m]
            return models[0] if models else ""
    except Exception:
        pass
    return ""


def _strip_think(chunks: Generator[str, None, None]) -> Generator[str, None, None]:
    """Loại bỏ khối <think>...</think> (một số model reasoning trả về trong content)."""
    in_think = False
    buf = ""
    for piece in chunks:
        buf += piece
        out = ""
        while buf:
            if in_think:
                end = buf.find("</think>")
                if end == -1:
                    buf = buf[-8:]
                    break
                buf = buf[end + 8 :]
                in_think = False
            else:
                start = buf.find("<think>")
                if start == -1:
                    keep = 7 if "<" in buf[-7:] else 0
                    out += buf[: len(buf) - keep] if keep else buf
                    buf = buf[len(buf) - keep :] if keep else ""
                    break
                out += buf[:start]
                buf = buf[start + 7 :]
                in_think = True
        if out:
            yield out
    if buf and not in_think:
        yield buf


def stream_groq(
    prompt: str, system_prompt: str, api_key: str, model: str, max_tokens: int = 750
) -> Generator[str, None, None]:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # Tự động xoay vòng qua các model Groq và hạ max_tokens dưới 1000 để không bao giờ bị lỗi OTPM Limit 1000
    candidate_models = [model] + [
        m for m in ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b") if m != model
    ]
    last_err = ""
    response = None
    for cand_model in candidate_models:
        for tok_limit in (max_tokens, 500):
            payload = {
                "model": cand_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": tok_limit,
                "stream": True,
            }
            resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=(20, 90))
            if resp.status_code == 200:
                response = resp
                break
            last_err = f"Groq ({cand_model}) HTTP {resp.status_code}: {resp.text[:220]}"
            if resp.status_code not in (400, 413, 429):
                break
        if response is not None:
            break

    if response is None:
        raise RuntimeError(last_err)
    response.encoding = "utf-8"

    def _gen():
        for line in response.iter_lines():
            if not line:
                continue
            s = line.decode("utf-8", errors="replace").strip()
            if s.startswith("data: ") and s != "data: [DONE]":
                try:
                    delta = json.loads(s[6:]).get("choices", [{}])[0].get("delta", {})
                except json.JSONDecodeError:
                    continue
                if delta.get("content"):
                    yield delta["content"]

    yield from _strip_think(_gen())


def stream_ollama(
    prompt: str, system_prompt: str, model: str
) -> Generator[str, None, None]:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": True,
        "options": {"temperature": 0.2, "repeat_penalty": 1.15, "num_predict": 768},
    }
    response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, stream=True, timeout=300)
    if response.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {response.status_code}: {response.text[:300]}")
    response.encoding = "utf-8"

    def _gen():
        for line in response.iter_lines():
            if not line:
                continue
            try:
                text = json.loads(line.decode("utf-8", errors="replace")).get("response", "")
            except json.JSONDecodeError:
                continue
            if text:
                yield text

    yield from _strip_think(_gen())


def generate_answer(
    user_prompt: str, system_prompt: str, llm: Dict[str, str]
) -> tuple[str, str]:
    """Sinh câu trả lời với backend đã chọn; tự động lùi Groq -> Ollama -> chỉ tra cứu.

    Trả về (full_response, backend_used).
    """
    attempts = []
    if llm.get("groq_model"):
        attempts.append(("Groq", lambda: stream_groq(user_prompt, system_prompt, llm["groq_key"], llm["groq_model"])))
    if llm.get("ollama_model"):
        attempts.append(("Ollama", lambda: stream_ollama(user_prompt, system_prompt, llm["ollama_model"])))

    errors = []
    for name, make_stream in attempts:
        try:
            with st.spinner(f"🧠 Đang soạn câu trả lời ({name})..."):
                gen = make_stream()
                first = next(gen, "")  # lỗi kết nối/HTTP sẽ nổ ra ở đây
            if not first:
                raise RuntimeError("phản hồi rỗng")

            def _chain(first_piece=first, rest=gen):
                yield first_piece
                yield from rest

            text = st.write_stream(_chain())
            if isinstance(text, str) and text.strip():
                return text, name
            raise RuntimeError("phản hồi rỗng")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")

    msg = (
        "**[Chế độ Tra cứu - Không LLM]**\n\n"
        "Dưới đây là các điều luật có độ khớp cao nhất được hệ thống tìm thấy dựa trên câu hỏi của bạn."
    )
    if errors:
        st.warning("Không gọi được mô hình ngôn ngữ — " + " | ".join(errors))
    st.markdown(msg)
    return msg, "Không LLM"


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

    corpus = load_corpus()
    laws_catalog = get_laws_catalog(corpus)
    bm25_retriever = load_bm25()
    hybrid_retriever, hybrid_error = load_hybrid(bm25_retriever)
    finetuned_reranker = load_finetuned_reranker()
    normalizer = load_normalizer()
    contextualizer = load_contextualizer()
    compressor = load_compressor()

    groq_key = get_secret_or_env("GROQ_API_KEY").strip().strip("'\"")
    groq_model, groq_error = pick_groq_model(groq_key)
    ollama_model = pick_ollama_model()
    llm = {"groq_key": groq_key, "groq_model": groq_model, "ollama_model": ollama_model}

    # --- SIDEBAR CONFIGURATION ---
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/scales.png", width=64)
        st.title("⚖️ Cấu Hình RAG")

        st.subheader("1. Phương Pháp Truy Xuất")
        if hybrid_retriever:
            retriever_options = [RETRIEVER_HYBRID, RETRIEVER_BM25]
        elif finetuned_reranker:
            retriever_options = [RETRIEVER_BM25_LORA, RETRIEVER_BM25]
        else:
            retriever_options = [RETRIEVER_BM25]
        retriever_choice = st.radio(
            "Retriever:",
            retriever_options,
            index=0,
            help="Chế độ chính xác nhất kết hợp bộ tái xếp hạng LoRA Legal Reranker đã fine-tune.",
        )
        if finetuned_reranker and retriever_choice in (RETRIEVER_HYBRID, RETRIEVER_BM25_LORA):
            st.success(f"✅ Đã kích hoạt LoRA Legal Reranker (α={finetuned_reranker.fusion_alpha})")
        top_k = st.slider("Số điều luật trích xuất (Top-K):", min_value=1, max_value=10, value=5)

        st.divider()
        st.subheader("2. Mô hình Ngôn ngữ (tự động)")
        if groq_model:
            st.success(f"✅ Groq Cloud — `{groq_model}`")
        elif ollama_model:
            st.info(f"✅ Ollama local — `{ollama_model}`")
            if groq_error:
                st.caption(f"Groq: {groq_error}")
        else:
            st.warning("⚠️ Không có LLM — chỉ hiển thị điều luật tra cứu được.")
            st.caption(f"Groq: {groq_error} | Ollama: không chạy.")
        if groq_model and ollama_model:
            st.caption(f"Dự phòng khi Groq lỗi: Ollama `{ollama_model}`")

        st.divider()
        st.caption(f"📊 Dữ liệu: **{len(corpus):,} điều luật** | **{len(laws_catalog):,} văn bản**")
        if st.button("🗑️ Xóa lịch sử hỏi đáp", width="stretch"):
            st.session_state.messages = []
            st.session_state.query_logs = []
            st.rerun()

    active_retriever = hybrid_retriever if retriever_choice == RETRIEVER_HYBRID else bm25_retriever
    active_reranker = (
        finetuned_reranker
        if retriever_choice in (RETRIEVER_HYBRID, RETRIEVER_BM25_LORA)
        else None
    )

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
                retrieved_docs = run_search(search_query, top_k, active_retriever, corpus, reranker=active_reranker)
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

                t_llm0 = time.time()
                full_response, backend_used = generate_answer(user_prompt, system_prompt, llm)
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
                    "Mô hình": backend_used,
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
        st.caption("Tìm kiếm dựa trên BM25 và Hybrid (BM25 + BGE-M3), không cần LLM.")

        c_query, c_method, c_k = st.columns([3, 1.5, 1])
        search_kw = c_query.text_input("Từ khóa hoặc nội dung điều luật:", placeholder="Ví dụ: thời gian thử việc, nghỉ thai sản, hợp đồng mùa vụ...")
        search_method = c_method.selectbox(
            "Phương pháp:",
            retriever_options,
            index=0,
            key="fast_search_method",
        )
        search_topk = c_k.number_input("Top-K:", min_value=1, max_value=20, value=5, step=1)

        if st.button("🚀 Bắt Đầu Tìm Kiếm", type="primary", width="stretch") or search_kw:
            if not search_kw.strip():
                st.info("Vui lòng nhập từ khóa để tìm kiếm.")
            else:
                norm_kw, acronyms, _ = normalizer.normalize(search_kw)
                if acronyms:
                    st.info(f"💡 Đã nhận diện và chuẩn hóa từ viết tắt: {', '.join([f'{k} -> {v}' for k, v in acronyms.items()])}")

                t_s0 = time.time()
                with st.spinner("Đang tìm kiếm..."):
                    search_retriever = hybrid_retriever if search_method == RETRIEVER_HYBRID else bm25_retriever
                    search_reranker = finetuned_reranker if search_method == RETRIEVER_HYBRID else None
                    results = run_search(norm_kw, int(search_topk), search_retriever, corpus, reranker=search_reranker)
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
            st.dataframe(log_df, width="stretch")

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
