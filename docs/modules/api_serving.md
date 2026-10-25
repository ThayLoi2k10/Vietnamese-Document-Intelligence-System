# Module: API Serving & Backend Layer (`src/api/`)

## 1. Tổng quan & Trách nhiệm (What does it do?)
Gói module [`src/api/`](file:///d:/PROJECT/PROJECT%202/src/api/) cung cấp tầng phục vụ Web API RESTful hiệu năng cao dựa trên nền tảng **FastAPI**:
* **Trách nhiệm chính:**
  * Tiếp nhận yêu cầu từ client (Web UI, Mobile app, Chatbot).
  * Điều phối các module bên dưới (Normalizer, Retrievers, Context Assembler, LLM Backend, Citation Verifier).
  * Quản lý trạng thái nạp bộ nhớ (in-memory caching & lazy-loading) của các mô hình và chỉ mục.
  * Hỗ trợ chuyển mạch linh hoạt nhiều backend suy luận LLM (Local GPU, Google Colab Tunnel, OpenAI-compatible server, hoặc Mock).

---

## 2. Bản đồ Cấu trúc Module (Package Map)

```text
src/api/
├── __init__.py           # Package exports
├── main.py               # Khởi tạo FastAPI app, cấu hình CORS, vòng đời lifespan
├── routes.py             # Định nghĩa Router (/api/v1), Dependency Injection (get_rag_service)
├── schemas.py            # Pydantic v2 Models: Request & Response schemas
└── service.py            # LegalRAGService (Singleton Service Container)
```

---

## 3. Các Class Cốt lõi & Kiến trúc Vận hành

### A. Tầng Điều Phối: `LegalRAGService` (`src/api/service.py`)
* Hoạt động theo mô hình **Singleton Container**: Khởi tạo một lần duy nhất khi ứng dụng chạy hoặc nạp lười (lazy-load) khi có request đầu tiên.
* Quản lý tài nguyên bộ nhớ:
  - `corpus`: DataFrame nạp từ `corpus_articles_preprocessed.parquet` (59.636 điều luật).
  - `bm25_retriever`: Inverted Index Okapi BM25 (`bm25_index.pkl`).
  - `dense_retriever`: Chỉ mục FAISS BGE-M3 (`dense_index_bge_m3.faiss`).
  - `reranker`: Cross-Encoder (`bge-reranker-base`).
  - `normalizer` & `contextualizer`: Bộ tiền xử lý câu hỏi.
  - `grounding_verifier`: Bộ thẩm định trích dẫn và căn cứ pháp lý.

### B. Cơ chế Hỗ trợ Đa Backend LLM (Multi-Backend Switching)
Tùy thuộc vào tham số `backend` trong request payload hoặc biến môi trường:
1. `colab` / `colab_tunnel`: Gửi HTTP POST tới URL công khai của ngrok/Cloudflare Tunnel kết nối với session Google Colab đang chạy `Qwen2.5-7B-Instruct`.
2. `local`: Nạp trực tiếp mô hình qua HuggingFace transformers vào GPU của máy chủ.
3. `openai_compatible`: Kết nối tới các runtime tăng tốc (vLLM, Ollama, TGI) thông qua chuẩn API OpenAI.
4. `mock`: Sinh câu trả lời giả lập cho môi trường CI/CD hoặc chạy Unit Test.

### C. Hợp đồng Dữ liệu Schema (`src/api/schemas.py`)
* **`SearchRequest`:** `query: str`, `retriever_type: Literal["bm25", "dense", "hybrid", "rerank"]`, `top_k: int = 10`.
* **`SearchResponse`:** `query`, `normalized_query`, `total_results`, `documents: list[RetrievedDocument]`, `latency_ms: float`.
* **`QueryRequest`:** `query: str`, `conversation_history: list[dict] = []`, `backend: str = "auto"`, `top_k: int = 5`, `temperature: float = 0.0`.
* **`QueryResponse`:** `query`, `answer`, `confidence`, `grounding_verdict`, `citations`, `retrieved_documents`, `latency_ms`.

---

## 4. Chi tiết các RESTful Endpoints (`src/api/routes.py`)

| Phương thức | Đường dẫn Endpoint | Mô tả chức năng | Tệp xử lý |
|---|---|---|---|
| `GET` | `/api/v1/health` | Kiểm tra trạng thái hệ thống, chỉ mục nạp và thời gian uptime | `routes.py:health_check` |
| `POST` | `/api/v1/search` | Tìm kiếm văn bản độc lập với các tùy chọn retriever đa dạng | `routes.py:search_documents` |
| `POST` | `/api/v1/query` | Hỏi đáp RAG End-to-End hoàn chỉnh kèm kiểm chứng căn cứ | `routes.py:answer_legal_query` |
| `GET` | `/api/v1/documents/{article_id}` | Lấy toàn văn nội dung và metadata của một điều luật theo ID | `routes.py:get_document_by_id` |

---

## 5. Hướng dẫn Khởi chạy & Kiểm thử Dịch vụ

### A. Khởi chạy Server cục bộ (Local Development)
```bash
# Khởi chạy FastAPI với uvicorn tại cổng 8000
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
* **Interactive Swagger UI:** Mở trình duyệt tại `http://localhost:8000/docs` để tương tác trực quan với các endpoints.
* **Redoc UI:** Mở tại `http://localhost:8000/redoc`.

### B. Chạy Bộ Kiểm thử Tự động (Automated API Tests)
```bash
# Chạy toàn bộ test suite cho API bằng pytest
pytest tests/test_api.py -v
```
Toàn bộ các trường hợp kiểm thử (Health check, Search với các retriever khác nhau, Query với mock backend, Tra cứu document theo ID hợp lệ/bất hợp lệ) đều được tự động xác thực bằng `TestClient`.


---

## Streamlit app (`app.py`) — cấu hình serving local đã kiểm thử (2026-10-25)

- **Retriever hiển thị cho người dùng (2 lựa chọn):**
  - *Hybrid (BM25 + BGE-M3)* — mặc định. `HybridRetriever` weighted RRF theo `data/indices/hybrid_config.json` (k_rrf=5, w_dense=0.75, w_bm25=0.25, depth=100); nDCG@10 = 0.6448 (Phase 11).
  - *BM25* — BM25 tuned (k1=1.80, b=0.85 được gán sau `load`), nDCG@10 = 0.4866 (Phase 5).
  - Dense-only và Hybrid + Reranker đã bị bỏ khỏi UI: Dense-only kém Hybrid, còn reranker zero-shot làm giảm nDCG@10 (0.6096) và chậm (~580 ms/query trên GPU).
- **Dense encoder dùng cho serving:** `src/retrieval/dense_hf_retriever.py` (`HFDenseRetriever`) dùng `transformers` thuần (CLS pooling + L2 normalize) trên `data/indices/dense_bge_m3/embeddings.f32.npy` có sẵn, không build lại. Lý do: `FlagEmbedding` import `datasets` → `pyarrow._dataset_parquet`, và DLL này bị Windows Application Control chặn trên máy dev. Đã kiểm tra: cosine giữa embedding encode lại và embedding đã lưu ≈ 1.000; truy vấn trên CPU ≈ 0.5 s.
- **LLM (tự chọn, không có ô nhập key):** Groq (`GROQ_API_KEY` lấy từ `.env`/secrets; model lấy từ danh sách live theo thứ tự `qwen/qwen3.8-27b` → `openai/gpt-oss-120b` → …) → Ollama local (nếu đang chạy) → chế độ chỉ tra cứu. Gemini và Colab tunnel đã bị bỏ khỏi UI.
- `.env` được đọc trực tiếp trong `app.py` (không phụ thuộc `python-dotenv`).
- Chạy: `.\.venv\Scripts\python.exe -m streamlit run app.py`
