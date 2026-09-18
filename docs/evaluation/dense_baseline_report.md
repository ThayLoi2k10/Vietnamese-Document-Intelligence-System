# Phase 6 — Baseline 2: Dense Retrieval (BGE-M3)

Tài liệu này ghi lại toàn bộ cơ sở lý thuyết, kiến trúc kỹ thuật, quy trình xây dựng chỉ mục và kết quả đối chuẩn thực nghiệm chuyên sâu của mô hình **Dense Bi-Encoder Retrieval (BGE-M3)** trên hệ thống văn bản pháp luật Việt Nam, đồng thời thực hiện phân tích đối chuẩn trực tiếp với mô hình từ khóa **BM25 Tuned** (Phase 5).

---

## 1. Tóm Tắt Kết Quả Cốt Lõi (Executive Summary)

Trong Phase 6, hệ thống đã triển khai thành công mô hình **Dense Retrieval (Zero-shot Bi-Encoder)** dựa trên pretrained model đa ngữ tiên tiến `BAAI/bge-m3` (revision `5617a9f61b028005a4858fdac845db406aefb181`). 

### Các phát hiện thực nghiệm quan trọng trên Hold-out Test (438 queries):
1. **Vượt trội toàn diện trên toàn bộ hệ thống metric IR:**
   * **nDCG@10** đạt **0.6277** (tăng **+29.00%** so với BM25 Tuned $0.4866$).
   * **MRR@10** đạt **0.6146** (tăng **+33.33%** so với BM25 Tuned $0.4610$).
   * **MAP@10** đạt **0.5520** (tăng **+34.18%** so với BM25 Tuned $0.4114$).
   * **Recall@10** đạt **0.7969** (tăng **+19.46%** so với BM25 Tuned $0.6671$).
   * **Recall@100** đạt **0.9441** (tăng **+8.18%** so với BM25 Tuned $0.8727$).
   * **Precision@1** đạt **0.4863** (tăng **+47.90%** so với BM25 Tuned $0.3288$).
2. **Khả năng giải cứu vượt trội các ca Hard Failures của từ khóa:**
   * Trong số 112 câu hỏi mà BM25 gặp thất bại hoàn toàn tại Top-10 ($R@10 = 0$), Dense Retrieval đã **giải cứu thành công 67 câu hỏi** (**Tỷ lệ giải cứu 59.82%**).
   * Đặc biệt, trong 67 ca giải cứu này, có **26 câu hỏi (38.81%)** được đưa thẳng lên vị trí **Rank 1**.
3. **Hiệu năng suy luận thực tế (Inference Latency & Throughput):**
   * Độ trễ trung bình: **19.10 ms / query** trên GPU NVIDIA GeForce MX350.
   * Thời gian tìm kiếm vector FAISS: **0.81 ms / query** trên CPU.
   * Thông lượng: **52.37 QPS**.

---

## 2. Thiết Kế Kiến Trúc & Ràng Buộc Hệ Thống (System Contracts)

Tuân thủ nghiêm ngặt các nguyên tắc bất biến của repository và đặc tả Phase 6:

* **Không gian ID bất biến:** Giữ nguyên vẹn toàn bộ **59,636 điều luật** (`global_article_id` trong khoảng liên tục $[0, 59635]$) từ `corpus_articles_preprocessed.parquet`. Đúng 8 điều luật rỗng được bảo tồn nguyên vẹn với nhãn `is_empty_preprocessed = True` và được mã hóa thành vector chuẩn tắc mà không bị xóa bỏ.
* **Mô hình cốt lõi:**
  * Model: `BAAI/bge-m3` (được khóa cố định theo commit revision `5617a9f61b028005a4858fdac845db406aefb181`).
  * Biểu diễn: Thuần Dense (`dense_only`), trích xuất từ native `[CLS]` token embedding, kích thước không gian vector $D = 1024$.
  * Chuẩn hóa: Chuẩn hóa $L_2$ ($||\mathbf{v}||_2 = 1.0$) cho mọi vector văn bản và truy vấn.
* **Chỉ mục tìm kiếm:** FAISS `IndexFlatIP` (Exact Inner Product Search). Do các vector đều được $L_2$-normalized, phép nhân vô hướng Inner Product đồng nhất tuyệt đối với Cosine Similarity:
  $$\langle \mathbf{u}, \mathbf{v} \rangle = \frac{\mathbf{u} \cdot \mathbf{v}}{||\mathbf{u}||_2 ||\mathbf{v}||_2} = \cos(\theta)$$
* **Giới hạn độ dài ngữ cảnh (Token-length Boundaries):**
  * Dựa trên phân tích kiểm toán `token_length_audit.py`:
    * Query max length: Khóa ở **512 tokens** (Tỷ lệ cắt cụt trên Train: 0.00%, độ dài dài nhất là 55 tokens).
    * Article max length: Khóa ở **2048 tokens** (Bao phủ 98.81% điều luật của toàn bộ corpus, chỉ 1.19% điều luật dài bị cắt cụt).
* **Bảo vệ tính đóng băng (Frozen Benchmark Protection):**
  * Tập Hold-out Test (`retrieval_test.json`, 438 queries) hoàn toàn không được sử dụng để tinh chỉnh tham số hay chọn lọc cấu hình.
  * Toàn bộ tham số được khóa cứng trước bằng `dense_pre_test_manifest.json` trước khi chạy nghiệm thu trên Test.

---

## 3. Bảng Đối Chuẩn Thực Nghiệm: BM25 Tuned vs. Dense BGE-M3

Bảng so sánh chi tiết hiệu năng trên tập kiểm thử độc lập **Hold-out Test Set (438 queries)**:

| Chỉ số (IR Metric) | BM25 Tuned ($k_1=1.8, b=0.85$) | Dense Retrieval (BGE-M3) | Chênh lệch Tuyệt đối (Delta) | Tỷ lệ Cải thiện (%) |
| :--- | :---: | :---: | :---: | :---: |
| **P@1** | 0.3288 | **0.4863** | +0.1575 | **+47.90%** |
| **P@5** | 0.1370 | **0.1685** | +0.0315 | **+22.99%** |
| **P@10** | 0.0817 | **0.0982** | +0.0165 | **+20.16%** |
| **R@1** | 0.2846 | **0.4176** | +0.1330 | **+46.73%** |
| **R@5** | 0.5644 | **0.7018** | +0.1374 | **+24.35%** |
| **R@10** | 0.6671 | **0.7969** | +0.1298 | **+19.46%** |
| **R@20** | 0.7468 | **0.8555** | +0.1087 | **+14.56%** |
| **R@50** | 0.8231 | **0.9177** | +0.0946 | **+11.49%** |
| **R@100** | 0.8727 | **0.9441** | +0.0714 | **+8.18%** |
| **MRR@10** | 0.4610 | **0.6146** | +0.1536 | **+33.33%** |
| **MAP@10** | 0.4114 | **0.5520** | +0.1406 | **+34.18%** |
| **MAP@100** | 0.4227 | **0.5611** | +0.1384 | **+32.75%** |
| **nDCG@10** | 0.4866 | **0.6277** | +0.1411 | **+29.00%** |
| **nDCG@100** | 0.5339 | **0.6634** | +0.1295 | **+24.25%** |
| **Latency / Query** | **3.57 ms** | 19.10 ms | +15.53 ms | — |
| **Throughput (QPS)** | **280.11 QPS** | 52.37 QPS | — | — |

> **Nhận xét chính:**
> Mô hình Dense Bi-Encoder BGE-M3 vượt trội mô hình từ khóa BM25 Tuned một cách áp đảo trên mọi chỉ số từ Top-1 đến Top-100. Độ chính xác tại vị trí đầu tiên ($P@1, R@1$) tăng gần 50%, cho thấy vector ngữ nghĩa định vị câu trả lời chính xác vượt trội hơn hẳn việc đếm trùng từ khóa.

---

## 4. Phân Tích Ma Trận Hiệu Quả & Khả Năng Giải Cứu (Rescue Analysis)

Nghiên cứu câu hỏi cốt lõi của Phase 6: *"Dense Retrieval có thực sự giải cứu được các câu hỏi mà BM25 gặp thất bại nặng nề do rào cản từ vựng hay không?"*

### A. Ma trận hiệu quả 4 kịch bản (Four-way Outcome Matrix)
Phân loại toàn bộ 438 queries trên Test Set dựa trên ngưỡng thành công $R@10 > 0$:

| Kịch bản | Số lượng câu hỏi | Tỷ lệ (%) | Ý nghĩa kỹ thuật |
| :--- | :---: | :---: | :--- |
| **1. Both Succeed (Cả hai cùng thành công)** | 314 | **71.69%** | Cả hai mô hình đều tìm thấy ít nhất 1 tài liệu liên quan trong Top-10. |
| **2. Dense Rescue (Dense giải cứu thành công)** | 67 | **15.30%** | BM25 hoàn toàn thất bại ($R@10=0$), nhưng Dense đưa tài liệu vào Top-10. |
| **3. BM25 Only (BM25 thành công, Dense thất bại)** | 12 | **2.74%** | BM25 tìm thấy trong Top-10, nhưng Dense trượt khỏi Top-10. |
| **4. Both Fail (Cả hai cùng thất bại)** | 45 | **10.27%** | Cả hai mô hình đều không tìm thấy tài liệu liên quan trong Top-10. |

### B. Tỷ lệ giải cứu đối với BM25 Hard Failures
* Tổng số ca BM25 Hard Failures ($R@10 = 0$): **112 queries** (chiếm 25.57% test set).
* Số ca được Dense giải cứu: **67 queries**.
* **Tỷ lệ giải cứu thành công (Rescue Rate):** $\frac{67}{112} = \mathbf{59.82\%}$.

### C. Phân bổ thứ hạng tài liệu được giải cứu (Rank Distribution)
Trong 67 câu hỏi được giải cứu, thứ hạng xuất hiện đầu tiên của tài liệu liên quan tập trung mạnh ở các vị trí đầu bảng:
* **Rank 1:** 26 queries (**38.81%**)
* **Rank 2:** 7 queries (**10.45%**)
* **Rank 3:** 4 queries (**5.97%**)
* **Rank 4:** 7 queries (**10.45%**)
* **Rank 5:** 8 queries (**11.94%**)
* **Rank 6:** 5 queries (7.46%)
* **Rank 7:** 2 queries (2.99%)
* **Rank 8:** 3 queries (4.48%)
* **Rank 9:** 1 query (1.49%)
* **Rank 10:** 4 queries (5.97%)

**Tổng cộng:** Hơn **67.1%** các ca giải cứu được xếp vào nửa trên của bảng kết quả (Top-1 đến Top-5).

---

## 5. Phân Tích Định Tính & Bản Chất Cơ Chế (Qualitative Error Analysis)

### A. Tại sao Dense Retrieval chiến thắng áp đảo (Dense Rescue Mechanism)?
1. **Khắc phục triệt để khoảng cách từ vựng (Vocabulary Mismatch / Paraphrase):**
   * *Ví dụ thực tế (Query 6184):* `"Tham ô giá trị bao nhiêu sẽ bị truy cứu trách nhiệm hình sự?"`
     * Ground Truth: Điều 353 Bộ luật Hình sự (Tội tham ô tài sản) — `article_id = 56797`.
     * **BM25:** $R@10 = 0.0$, $R@100 = 0.0$ (thất bại toàn diện vì câu hỏi dùng từ ngữ đời thường, trong khi văn bản luật dùng cấu trúc định lượng phức tạp).
     * **Dense BGE-M3:** Xếp đúng điều luật này tại **Rank 1** ($P@1 = 1.0, MRR@10 = 1.0, nDCG@10 = 1.0$) với điểm tương đồng $0.7725$.
2. **Hiểu ngữ nghĩa suy diễn hành vi pháp lý (Legal Semantic Abstraction):**
   * *Ví dụ thực tế (Query 2311):* `"Việc sử dụng trang phục y tế được quy định như thế nào?"`
     * Ground Truth: `article_id = 16136`.
     * **BM25:** $R@10 = 0.0$ (bị nhiễu bởi các điều luật về "trang phục" của các ngành khác như hải quan, công an).
     * **Dense BGE-M3:** Đưa đúng điều luật vào **Rank 1** ($MRR@10 = 1.0$).
3. **Phát hiện quan hệ tương đương (Synonym & Semantic Expansion):**
   * *Ví dụ thực tế (Query 6205):* `"Bán xe do người khác ăn trộm được có bị truy cứu hình sự?"`
     * Ground Truth: Điều 323 Bộ luật Hình sự (Tội chứa chấp hoặc tiêu thụ tài sản do người khác phạm tội mà có) — `article_id = 56767`.
     * Cụm từ "bán xe ăn trộm" và "tiêu thụ tài sản do người khác phạm tội mà có" không có một từ vựng pháp lý nào trùng lặp. BM25 hoàn toàn bỏ lỡ ($R@100 = 0.0$). Ngược lại, Dense bắt được mối liên hệ ngữ nghĩa tiềm ẩn này và đưa vào danh sách ứng viên ($R@50 = 1.0$).

### B. Tại sao có 12 trường hợp BM25 thành công nhưng Dense thất bại (BM25 Only)?
Khi phân tích 12 trường hợp này, các nguyên nhân kỹ thuật bộc lộ:
1. **Truy vấn chứa thuật ngữ đặc thù chuyên biệt hoặc số hiệu định danh:**
   * Khi truy vấn hỏi đích danh mã văn bản, tên viết tắt của một thủ tục hành chính chuyên biệt, hoặc các thuật ngữ hiếm chỉ xuất hiện duy nhất ở một điều luật, BM25 tận dụng IDF cực cao để ghim tài liệu lên đầu bảng.
   * Bi-encoder nén toàn bộ văn bản dài 2,048 tokens vào một vector 1,024 chiều, dẫn đến hiện tượng pha loãng thông tin (Information Dilution), làm mờ các token từ khóa đặc thù này.
2. **Mật độ từ khóa cao (Keyword Density):**
   * Các truy vấn trích dẫn nguyên văn một câu trong luật khiến BM25 cho điểm số vượt trội do khớp chính xác từng từ (Exact String Match).

### C. Tại sao có 45 trường hợp cả hai cùng thất bại (Both Fail)?
1. **Suy luận pháp lý đa bước (Multi-hop Reasoning):**
   * Câu hỏi yêu cầu tổng hợp điều kiện từ điều luật chung dẫn chiếu sang điều luật chuyên ngành (ví dụ: dẫn chiếu giữa Bộ luật Dân sự và Luật Đất đai).
2. **Ngữ cảnh quá rộng hoặc mơ hồ:**
   * Câu hỏi người dùng quá ngắn, thiếu định ngữ pháp lý xác định (ví dụ: hỏi chung chung về "thời hạn giải quyết", "hồ sơ gồm những gì").
3. **Giới hạn của Zero-shot Pretrained Encoder:**
   * `BAAI/bge-m3` là mô hình tổng quát (General Domain), chưa qua bước Domain Adaptation hay Fine-tuning có giám sát trên các cặp câu hỏi - điều luật của Việt Nam (sẽ là cơ hội cải thiện ở các pha sau).

---

## 6. Hiệu Năng & Chi Phí Tài Nguyên Hệ Thống (System Efficiency)

### A. Thời gian xây dựng chỉ mục (Index Build)
* **Quy mô Corpus:** 59,636 điều luật.
* **Cơ chế tối ưu:** Dynamic Bucketed Batching theo độ dài (`short: 32, medium: 16, long: 8, very_long: 4`) + Checkpointing từng chunk 2,000 điều luật.
* **Thời gian mã hóa toàn bộ:** ~276 phút trên 1 GPU NVIDIA GeForce MX350 (2GB VRAM, CUDA FP16).
* **Bộ nhớ kích hoạt:** Đỉnh điểm < 1,250 MB VRAM (hoàn toàn loại bỏ hiện tượng tràn bộ nhớ CUDA OOM).
* **Kích thước chỉ mục:**
  * `embeddings.f32.npy`: **244.27 MB** ($59,636 \times 1,024$ phần tử Float32).
  * `index.faiss`: **244.27 MB** (Flat IP index binary).
  * Metadata & Fingerprints: ~0.6 MB.
  * Tổng dung lượng lưu trữ: **~489 MB**.

### B. Độ trễ truy vấn (Query Latency & Throughput)
* Thời gian nạp Retriever vào RAM/VRAM: **9.34s**.
* Thời gian mã hóa câu hỏi: **18.28 ms / query** (chạy batch trên GPU).
* Thời gian quét chỉ mục FAISS Flat IP: **0.81 ms / query** (chạy trên CPU).
* Tổng độ trễ truy xuất: **19.10 ms / query**.
* Thông lượng hệ thống: **52.37 QPS**.

---

## 7. Kiểm Định Cổng Đảm Bảo Chất Lượng (Validation Gates & QA)

Toàn bộ các tiêu chí nghiệm thu của Phase 6 đã được xác thực độc lập:

1. **Kiểm định tính toàn vẹn Corpus (Corpus Gate):**
   * Chạy script `scripts/validation/validate_preprocessed_corpus.py` -> **PASS 100%**.
   * Số dòng: 59,636; ID liên tục $[0, 59635]$; 8 điều luật rỗng được bảo toàn; text NFC.
2. **Kiểm định tính bất biến Benchmark (Benchmark Gate):**
   * Chạy script `scripts/validation/validate_retrieval_benchmark.py` -> **PASS 100%**.
   * Test set: 438 queries; SHA-256 khớp tuyệt đối với Phase 4 (`d74cf9dd085973a5...`); 0% data leakage.
3. **Bộ Unit Tests tự động (Unit Test Suite):**
   * Chạy lệnh `python -m unittest discover -s tests -p "test_*.py"` -> **26/26 tests PASS** (11.6s).
   * Bao gồm kiểm tra logic index FAISS, chuẩn hóa vector, cosine oracle comparison, artifact verification, deterministic encoding.
4. **Xác thực toàn vẹn Artifacts:**
   * Hàm `DenseArtifactManager.verify_artifacts()` thực hiện kiểm tra mã băm SHA-256, shape vector `[59636, 1024]`, kiểu dữ liệu `float32`, và sự hiện diện của 6/6 file thành phần -> **PASS 100%**.
5. **Thử nghiệm tương tác dòng lệnh (CLI Sanity):**
   * `scripts/retrieval/run_dense_retrieval.py` thực hiện tìm kiếm chính xác và trả về trích đoạn pháp lý định dạng chuẩn.

---

## 8. Kết Luận & Định Hướng Sang Phase 11 (Hybrid Retrieval)

Kết quả thực nghiệm của Phase 6 khẳng định vai trò nền tảng của **Dense Retrieval**:
1. Dense Retrieval phá vỡ hoàn toàn "bức tường từ vựng" (Lexical Chasm) của BM25, nâng nDCG@10 từ 0.4866 lên 0.6277 (+29%) và giải cứu gần 60% các ca thất bại nặng nề.
2. Tuy nhiên, sự tồn tại của 12 ca "BM25 Only" (nơi BM25 thành công nhờ từ khóa chuyên biệt, mã số, và trích dẫn chuẩn xác trong khi Dense bị phân tán) chứng minh rằng: **Mô hình từ khóa và mô hình vector biểu diễn hai góc nhìn bổ trợ hoàn hảo cho nhau (Complementary Strengths)**.
3. Đây là bằng chứng thực nghiệm vững chắc để triển khai **Phase 11 (Hybrid Retrieval: BM25 + Dense Fusion)** qua thuật toán Reciprocal Rank Fusion (RRF) hoặc Score Normalization kết hợp, hứa hẹn sẽ tối ưu hóa nDCG@10 lên các mốc cao hơn nữa trước khi bước vào Reranking và RAG.

---

## 9. Danh Mục Artifacts Sinh Ra Trong Phase 6

| Đường dẫn Artifact | Phân loại | Mô tả nội dung |
| :--- | :--- | :--- |
| `data/indices/dense_bge_m3/config.json` | Chỉ mục | Cấu hình kỹ thuật đầy đủ của mô hình và FAISS index |
| `data/indices/dense_bge_m3/corpus_ids.json` | Chỉ mục | Danh sách 59,636 ID bài viết ánh xạ tuần tự |
| `data/indices/dense_bge_m3/corpus_fingerprint.json` | Chỉ mục | Mã băm định danh toàn vẹn của Corpus và Model |
| `data/indices/dense_bge_m3/embeddings.f32.npy` | Chỉ mục | Ma trận vector 59,636 x 1024 (Float32, L2-normalized) |
| `data/indices/dense_bge_m3/index.faiss` | Chỉ mục | File nhị phân chỉ mục FAISS Flat IP |
| `data/indices/dense_bge_m3/build_report.json` | Chỉ mục | Báo cáo chi tiết thời gian và thông số build |
| `results/benchmarks/dense_pre_test_manifest.json` | Đóng băng | Bản kê khai thông số đã khóa trước khi test |
| `results/benchmarks/dense_train_metrics.json` | Đo lường | Toàn bộ 14 chỉ số IR trên tập Train (1,752 queries) |
| `results/benchmarks/dense_train_predictions.json` | Dự đoán | Chi tiết Top-100 xếp hạng cho 1,752 queries Train |
| `results/benchmarks/dense_train_vs_bm25_comparison.json` | Đối chuẩn | Bảng so sánh với BM25 trên tập Train |
| `results/benchmarks/dense_test_metrics.json` | Đo lường | Toàn bộ 14 chỉ số IR chính thức trên Test (438 queries) |
| `results/benchmarks/dense_test_predictions.json` | Dự đoán | Chi tiết Top-100 xếp hạng cho 438 queries Test |
| `results/benchmarks/dense_test_vs_bm25_comparison.json` | Đối chuẩn | Bảng so sánh chính thức giữa BM25 Tuned và Dense |
| `results/benchmarks/dense_vs_bm25_hard_failure_analysis.json` | Phân tích | Ma trận 4 nhóm, tỷ lệ giải cứu, mẫu phân tích định tính |
| `docs/evaluation/dense_baseline_report.md` | Tài liệu | Báo cáo khoa học và kỹ thuật hoàn chỉnh của Phase 6 |
