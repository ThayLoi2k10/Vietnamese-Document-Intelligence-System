# ⚖️ Vietnamese Legal Document Intelligence & RAG System

Hệ thống Trí tuệ Nhân tạo Tra cứu, Tái xếp hạng và Hỏi đáp Pháp luật Việt Nam (**End-to-End Vietnamese Legal Information Retrieval & Retrieval-Augmented Generation System**), được thiết kế và đánh giá thực nghiệm trên kho dữ liệu **59,636 điều luật** quy phạm pháp luật Việt Nam.

---

## 🌟 Điểm Nhấn Kỹ Thuật & Đóng Góp Cốt Lõi

1. **Chuẩn hóa & Làm phẳng Kho Dữ liệu Pháp luật (59,636 Điều luật):**
   - Ánh xạ toàn bộ văn bản quy phạm pháp luật về không gian định danh liên tục `global_article_id` $\in [0, 59635]$, chuẩn hóa Unicode dựng sẵn (NFC) và bảo toàn cấu trúc phân cấp *Điều $\to$ Khoản $\to$ Điểm*.
   - Xây dựng bộ đối chuẩn **Frozen Retrieval Benchmark** (`1,752` train / `438` test queries, phân tầng đơn nhãn & đa nhãn) và **QA Benchmark** (`29,145` cặp hỏi đáp phân tách theo nhóm `content_hash` MD5 chống rò rỉ ngữ cảnh).
2. **Truy xuất Hỗn hợp Hai Tầng (Hybrid Weighted RRF + Domain-Specific LoRA Reranker):**
   - **Tầng 1 — Hybrid Retrieval (`HybridRetriever`):** Kết hợp từ khóa **Okapi BM25 Tuned** ($k_1=1.80, b=0.85$) và ngữ nghĩa **Dense Bi-Encoder (`BAAI/bge-m3` 1024-d FAISS)** qua giải thuật *Weighted Reciprocal Rank Fusion* ($k_{\text{rrf}}=5, w_{\text{dense}}=0.75, w_{\text{bm25}}=0.25$), giải cứu **59.8%** các truy vấn thất bại từ vựng của BM25 và đạt `nDCG@10 = 0.6448`.
   - **Tầng 2 — Fine-tuned LoRA Legal Reranker (`FinetunedLegalReranker`):** Khai thác **9,807 mẫu âm khó trong miền (In-Domain Hard Negatives)** từ Top-20 Hybrid RRF trên tập `dev_train` (`1,401` queries) và huấn luyện tinh chỉnh **LoRA ($r=16, \alpha=32$)** trên toàn bộ 72 tầng tuyến tính của `BAAI/bge-reranker-base` bằng hàm mất mát kép (*ListNet Cross-Entropy + Pairwise Margin Ranking Loss*). Kết hợp cơ chế **Residual Hybrid Fusion ($\alpha = 0.45$)** giúp khắc phục hoàn toàn hiện tượng *Pool Depth Inversion*, nâng **Hit@1 lên 57.76%**, **Hit@5 lên 84.47%**, và **nDCG@10 lên 0.6911**.
3. **Đường ống RAG Kiểm chứng Trích dẫn & Chống Ảo giác (`src/rag/`):**
   - Tích hợp chuẩn hóa 48+ từ viết tắt pháp lý (`LegalAcronymNormalizer`), xử lý hội thoại nhiều lượt (`ConversationContextualizer`), nén ngữ cảnh theo cấp Khoản (`LegalPassageCompressor`), và bộ hậu kiểm trích dẫn tự động (`GroundingVerifier` & `AbstentionPolicy`) đạt **88.2% Citation Fidelity** và **85.0% End-to-End QA Accuracy**.

---

## 📊 Bảng Kết Quả Thực Nghiệm Chính Thức

### 1. Đánh giá Truy xuất & Tái xếp hạng trên Tập Frozen Test (`438` truy vấn)

| Hệ thống / Phương pháp | Hit@1 (%) | Hit@3 (%) | Hit@5 (%) | Hit@10 (%) | Recall@5 (%) | Recall@10 (%) | MRR@10 | MAP@10 | nDCG@10 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. TF-IDF Unigram (Phase 5)** | 28.08% | 46.12% | 55.25% | 65.07% | 46.89% | 56.92% | 0.3917 | 0.3511 | 0.4152 |
| **2. Okapi BM25 Tuned (Phase 5)** | 32.88% | 56.16% | 63.70% | 74.43% | 55.44% | 66.71% | 0.4610 | 0.4150 | 0.4866 |
| **3. Dense Bi-Encoder BGE-M3 (Phase 6)** | 48.63% | 69.86% | 79.00% | 86.99% | 70.66% | 79.96% | 0.6146 | 0.5629 | 0.6277 |
| **4. Hybrid Weighted RRF (Phase 11)** | 50.68% | 73.52% | 81.28% | 87.67% | 72.73% | 80.13% | 0.6366 | 0.5831 | 0.6448 |
| **5. Hybrid + Reranker Zero-shot (Phase 12)** | 43.84% | 64.84% | 72.15% | 84.47% | 63.96% | 76.59% | 0.5656 | 0.5117 | 0.5830 |
| **6. Hybrid + Fine-tuned LoRA Reranker (Phase 15)** | **57.76%** | **77.85%** | **84.47%** | **89.95%** | **76.29%** | **82.22%** | **0.6902** | **0.6364** | **0.6911** |

### 2. Đánh giá Hỏi đáp Pháp lý End-to-End (`results/benchmarks/end_to_end_qa_accuracy_report.json`)

| Chỉ số Đánh giá (End-to-End RAG Evaluation) | Kết quả (%) | Ghi chú Kỹ thuật |
|---|:---:|---|
| **Context Retrieval Hit@5** | **100.0%** | Tỷ lệ tìm đúng văn bản luật mục tiêu trong Top-5 ngữ cảnh đưa vào LLM |
| **End-to-End QA Accuracy** | **85.0%** | Tỷ lệ câu trả lời khớp chính xác căn cứ pháp lý và nội dung điều khoản |
| **Grounding Citation Fidelity** | **88.2%** | Tỷ lệ trích dẫn điều luật được `GroundingVerifier` xác thực `VERIFIED` |

---

## 🚀 Hướng Dẫn Khởi Chạy Nhanh (Quickstart)

### Cách 1: Chạy bằng Docker (Khuyên dùng — Đóng gói sẵn toàn bộ môi trường)

1. Tạo file `.env` từ `.env.example` và điền khóa `GROQ_API_KEY` (hoặc bật sẵn ứng dụng **Ollama** trên máy):
   ```bash
   cp .env.example .env
   ```
2. Khởi chạy container bằng 1 lệnh duy nhất:
   ```bash
   docker compose up --build
   ```
   *(Nếu máy mới clone repo chưa có sẵn file vector `index.faiss`, ứng dụng sẽ tự động tải về vào thư mục `./data/indices/dense_bge_m3/` ở lần khởi động đầu tiên).*
3. Mở trình duyệt tại: **`http://localhost:8501`**

### Cách 2: Chạy trực tiếp trên môi trường Python Local

```powershell
# 1. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# 2. Khởi chạy ứng dụng Web Streamlit
streamlit run app.py
```

---

## 🗂️ Cấu Trúc Thư Mục Dự Án

```text
PROJECT 2/
├── app.py                        # Ứng dụng Web Streamlit (Hỏi đáp RAG, Tra cứu nhanh, Thư viện Luật, Analytics)
├── Dockerfile                    # Cấu hình đóng gói Docker Image chuẩn sản xuất
├── docker-compose.yml            # Cấu hình khởi chạy dịch vụ kèm persistent cache volume
├── requirements.txt              # Danh sách thư viện phụ thuộc Python
├── configs/                      # Cấu hình YAML cho các thực nghiệm, datasets và versioned prompts
├── src/                          # Thư viện lõi của hệ thống
│   ├── retrieval/                # BM25, TF-IDF, Dense BGE-M3 (HFDenseRetriever), Hybrid wRRF, Chunking, Qdrant, LoRA Reranker
│   ├── rag/                      # Query Normalizer/Contextualizer, Context Compressor, Prompting, Grounding & Abstention
│   ├── evaluation/               # Các độ đo IR chuẩn mực (Precision@k, Recall@k, MRR@10, MAP@10, nDCG@10)
│   └── api/                      # Dịch vụ RESTful API trên nền FastAPI (/health, /search, /query, /documents)
├── scripts/                      # Các kịch bản thực thi theo từng giai đoạn
│   ├── data_acquisition/         # Tải dữ liệu từ Hugging Face
│   ├── processing/               # Làm phẳng corpus & tiền xử lý Unicode NFC / khoảng trắng cấu trúc
│   ├── validation/               # Các cổng kiểm định toàn vẹn schema, ID và chống rò rỉ dữ liệu
│   ├── benchmarks/               # Xây dựng tập đối chuẩn Retrieval & QA
│   ├── retrieval/                # Huấn luyện chỉ mục từ khóa, vector, chunking và tinh chỉnh siêu tham số
│   ├── indexing/                 # Xây dựng và kiểm định cơ sở dữ liệu vector Qdrant
│   ├── training/                 # Đào Hard Negatives & Fine-tune LoRA Reranker qua GPU Tunnel
│   └── evaluation/               # Đánh giá định lượng trên tập Frozen Test và End-to-End QA
├── notebooks/                    # Notebook khởi tạo GPU Worker trên Google Colab & chạy thực nghiệm
├── data/                         # Kho dữ liệu, bộ đối chuẩn và chỉ mục (bao gồm trọng số LoRA đã fine-tune)
├── results/benchmarks/           # Toàn bộ báo cáo JSON metrics & predictions thực nghiệm qua các Phase
├── artifacts/                    # Schemas JSON hợp đồng dữ liệu và báo cáo phân tích đối chứng Phase 9-10
├── tests/                        # Bộ kiểm thử tự động (Unit, Integration, Data Leakage tests)
└── docs/                         # Hệ thống tài liệu kỹ thuật chi tiết theo từng Phase & từng Module
```

---

## 📚 Tài Liệu Kỹ Thuật Chi Tiết (`docs/`)

Xem mục lục đầy đủ tại [**`docs/README.md`**](docs/README.md):
- **Theo tiến trình nghiên cứu & phát triển (Phase 1 $\to$ Phase 15):** [`docs/phases/`](docs/phases/)
- **Theo kiến trúc từng Module mã nguồn:** [`docs/modules/`](docs/modules/)
- **Báo cáo phân tích & đánh giá thực nghiệm:** [`docs/evaluation/`](docs/evaluation/)
