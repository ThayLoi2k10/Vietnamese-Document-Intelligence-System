# Hệ Thống Tài Liệu Kỹ Thuật (Vietnamese Legal Document Intelligence & RAG System)

Chào mừng bạn đến với trung tâm tài liệu kỹ thuật của dự án. Toàn bộ tài liệu được chuẩn hóa theo phong cách **Code-first, Architecture-first, No-fluff**: trực diện, cô đọng, có sơ đồ luồng dữ liệu, trích đoạn code và số liệu đối chuẩn thực nghiệm chính xác.

---

## 1. Danh Mục Tài Liệu Theo Tiến Trình Dự Án (`docs/phases/`)

| Phase | Tên giai đoạn | Mục tiêu cốt lõi | Tài liệu chi tiết |
|---|---|---|---|
| **Phase 1** | **Dataset Understanding & EDA** | Kiểm toán 3 luồng thô: 170k HTML docs, 29k QA, 59.636 điều luật | [Xem tài liệu](phases/phase01_dataset_understanding.md) |
| **Phase 2 & 3** | **Preprocessing & Corpus Flattening** | Làm phẳng corpus 59.636 điều, chuẩn hóa Unicode NFC, giữ cấu trúc | [Xem tài liệu](phases/phase02_03_preprocessing_and_flattening.md) |
| **Phase 4** | **Benchmark Split Construction** | Phân chia chống rò rỉ: Frozen Test (438 queries) & QA Group MD5 | [Xem tài liệu](phases/phase04_benchmark_construction.md) |
| **Phase 5** | **Lexical Baselines (TF-IDF & BM25)** | Lập chỉ mục từ khóa, tinh chỉnh BM25 ($k_1=1.8, b=0.85$, nDCG=0.4866) | [Xem tài liệu](phases/phase05_lexical_baselines.md) |
| **Phase 6** | **Dense Bi-Encoder (BGE-M3)** | Vector hóa ngữ nghĩa 1024-d, giải cứu 59.8% thất bại từ khóa | [Xem tài liệu](phases/phase06_dense_retrieval.md) |
| **Phase 7** | **Document Chunking Strategy** | Phân cấp cấu trúc Điều $\to$ Khoản $\to$ Điểm (79.977 chunks C9) | [Xem tài liệu](phases/phase07_document_chunking.md) |
| **Phase 8** | **Vector Database Indexing** | Qdrant Persistent Store, UUID v5 tất định, Recall=1.0000 vs Oracle | [Xem tài liệu](phases/phase08_vector_database_indexing.md) |
| **Phase 9** | **Modular RAG Architecture** | Kiến trúc Enterprise trong `src/rag/`, tách biệt 11 gói mô-đun | [Xem tài liệu](phases/phase09_rag_code_architecture.md) |
| **Phase 10** | **LLM-only vs RAG Experiment** | Đối chứng 1.401 câu: RAG dập tắt 100% tiếng Trung, đập tan tự tin ảo | [Xem tài liệu](phases/phase10_controlled_experiment_llm_vs_rag.md) |
| **Phase 11** | **Hybrid Retrieval (Weighted RRF)** | Production Retriever kết hợp BM25 + BGE-M3 (nDCG@10 = 0.6448) | [Xem tài liệu](phases/phase11_hybrid_retrieval.md) |
| **Phase 12** | **Cross-Encoder Reranking** | Đánh giá Cross-Encoder, phân tích Negative Dilution & độ trễ GPU | [Xem tài liệu](phases/phase12_cross_encoder_reranking.md) |
| **Phase 13** | **Query Processing Pipeline** | Normalizer 48+ từ viết tắt, Qwen2.5 Rewriter, Information Loss Gate | [Xem tài liệu](phases/phase13_query_processing.md) |
| **Phase 14** | **FastAPI Serving & Backend** | Đóng gói REST API `/health`, `/search`, `/query`, sẵn sàng Docker | [Xem tài liệu](phases/phase14_api_serving_architecture.md) |

---

## 2. Danh Mục Tài Liệu Theo Module Mã Nguồn (`docs/modules/`)

Dành cho lập trình viên và kỹ sư phần mềm tra cứu trực tiếp kiến trúc, class, phương thức và code ví dụ:

1. [**`retrieval_engine.md`**](modules/retrieval_engine.md) (`src/retrieval/`):
   - `BaseRetriever`, `RetrievalResult`, `BM25Retriever`, `TfidfRetriever`.
   - `DenseRetriever` (BGE-M3 FAISS), `HybridRetriever` (Weighted RRF).
   - Bộ phân đoạn cấu trúc `HierarchicalChunker` và cơ sở dữ liệu `QdrantVectorStore`.
2. [**`rag_pipeline.md`**](modules/rag_pipeline.md) (`src/rag/`):
   - `RAGPipeline` Orchestrator điều phối tuần tự từ Query đến Answer.
   - `ContextAssembler`, `BudgetEnforcer` (ngưỡng 2048 tokens), `Deduplicator`.
   - `PromptBuilder` nạp template phiên bản, `HuggingFaceGenerator` (`Qwen2.5-7B-Instruct`).
   - `OutputParser` bóc tách JSON và ánh xạ Pydantic `LegalAnswer`.
3. [**`guardrails_and_evaluation.md`**](modules/guardrails_and_evaluation.md) (`src/rag/abstention/`, `citation/`, `src/evaluation/`):
   - `GroundingVerifier` nhận diện 19 bộ luật và regex số hiệu văn bản quy phạm pháp luật.
   - `AbstentionPolicy` cơ chế chủ động từ chối trả lời an toàn khi thiếu căn cứ.
   - Bộ đo lường chất lượng truy xuất IR chuẩn mực (P@k, R@k, MRR@10, MAP@10, nDCG@10).
4. [**`query_processor.md`**](modules/query_processor.md) (`src/rag/query/`):
   - `LegalAcronymNormalizer` chuẩn hóa 48+ từ viết tắt chuyên ngành.
   - `ConversationContextualizer` xử lý câu hỏi tiếp nối và đại từ trong hội thoại nhiều lượt.
   - `QueryProcessor` tích hợp cổng kiểm soát mất thông tin (Information Loss Gate $< 50\%$).
5. [**`api_serving.md`**](modules/api_serving.md) (`src/api/`):
   - Ứng dụng FastAPI, middleware CORS, dependency injection `get_rag_service`.
   - 4 Endpoints: `GET /health`, `POST /search`, `POST /query`, `GET /documents/{id}`.
   - Cơ chế chuyển mạch đa backend: `local`, `colab_tunnel`, `openai_compatible`, `mock`.
