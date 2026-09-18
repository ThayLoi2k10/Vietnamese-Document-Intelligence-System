# Kỹ Thuật Benchmark & Đánh Giá Hệ Thống (Evaluation Dataset Documentation)

Tài liệu này đặc tả chi tiết thiết kế, phương pháp luận phân chia (Splitting Methodology), hợp đồng dữ liệu (Data Contract) và kết quả kiểm định toán học cho hai bộ dữ liệu đánh giá: **Retrieval Benchmark** và **QA Benchmark**.

---

## 1. Nguồn Gốc Dữ Liệu (Data Provenance)

Hệ thống đánh giá được xây dựng dựa trên 3 nguồn dữ liệu đã qua tiền xử lý chuẩn hóa Unicode NFC và cấu trúc khoảng trắng ở Phase 3:

1. **Corpus Văn Bản Pháp Luật (`corpus_articles_preprocessed.parquet`)**:
   - Quy mô: 59,636 articles.
   - Miền định danh (`global_article_id`): Dải số nguyên liên tục $[0, 59635]$.

2. **Retrieval Queries (`data/raw/legal_retrieval/train.json`)**:
   - Quy mô: 2,190 câu hỏi thực tế của người dùng kèm nhãn các điều khoản liên quan.
   - *Lưu ý*: Hai tập `public_test.json` (312 queries) và `private_test.json` (627 queries) bị ẩn ground-truth (`relevant_laws: []`) nên không dùng để tự động tính metric IR; toàn bộ 2,190 queries có nhãn được dùng để phân chia Train/Test.

3. **Legal QA (`data/processed/legal_qa/legal_qa_preprocessed.parquet`)**:
   - Quy mô: 29,145 cặp Question-Answer phân bổ trên 9,715 `doc_id`.
   - Đặc điểm: Mỗi tài liệu có đúng 3 câu hỏi đi kèm ($3 \times 9,715 = 29,145$).

---

## 2. Phương Pháp Luận Phân Chia (Splitting Methodology)

### 2.1. Retrieval Benchmark (Stratified 80/20 Split)
- **Vấn đề**: Trong 2,190 queries, có 75.1% truy vấn dạng Single-label (chỉ cần 1 điều khoản) và 24.9% dạng Multi-label (cần nhiều điều khoản, tối đa 9 điều khoản). Nếu chia ngẫu nhiên thuần túy, tỷ lệ multi-label có thể lệch giữa Train và Test, làm sai lệch các metric xếp hạng (MAP, nDCG@k).
- **Giải pháp**: Phân tầng (`StratifiedSplit`) theo nhãn nhị phân `is_multi_label` với `random_state=42`.
- **Dung lượng**:
  - **Train**: 1,752 queries (80.0%) — Multi-label chiếm 24.83%.
  - **Test**: 438 queries (20.0%) — Multi-label chiếm 24.89%.
- **Chống rò rỉ**: $\text{Train} \cap \text{Test} = \emptyset$ trên không gian `query_id`.

### 2.2. QA Benchmark (Stratified Group Split theo Context Hash)
- **Vấn đề (Data Leakage)**: Phân tích thực nghiệm phát hiện **130 nội dung điều khoản (`article_content`) bị dùng chung giữa 263 `doc_id` khác nhau** (do các điều khoản chuyển tiếp, hiệu lực hoặc quy định mẫu lặp lại qua các văn bản luật). Nếu chỉ group theo `doc_id`, các câu hỏi thuộc cùng một ngữ cảnh sẽ nằm ở cả Train và Test $\rightarrow$ Mô hình LLM/RAG bị rò rỉ ngữ cảnh kiểm thử.
- **Giải pháp**:
  1. Tính mã băm MD5 cho từng ngữ cảnh đã chuẩn hóa: `content_hash = md5(context.strip())`. Tổng số ngữ cảnh thực tế là **9,582 unique contexts**.
  2. Gom cụm theo `content_hash` để thực hiện phân hoạch cố định 3 tập Train (70%), Val (15%), Test (15%) với `random_state=42`.
  3. Phân tầng theo `strata = difficulty + "_" + question_type` để bảo toàn cấu trúc phân bố loại câu hỏi và độ khó.
- **Dung lượng**:
  - **Train**: 20,403 QA (70.01%) trên 6,708 contexts.
  - **Val**: 4,374 QA (15.01%) trên 1,439 contexts.
  - **Test**: 4,368 QA (14.99%) trên 1,435 contexts.
- **Chứng minh toán học Zero Leakage**:
  $$\text{Contexts}_{\text{train}} \cap \text{Contexts}_{\text{val}} = \emptyset$$
  $$\text{Contexts}_{\text{train}} \cap \text{Contexts}_{\text{test}} = \emptyset$$
  $$\text{Contexts}_{\text{val}} \cap \text{Contexts}_{\text{test}} = \emptyset$$

---

## 3. Đặc Tả Hợp Đồng Dữ Liệu (Data Contracts)

### 3.1. Retrieval Benchmark Schema (`data/benchmarks/retrieval_{train,test}.json`)

| Trường | Kiểu dữ liệu | Bắt buộc | Mô tả & Ràng buộc |
| :--- | :--- | :--- | :--- |
| `query_id` | `int` | Yes | ID duy nhất toàn cục |
| `query_text` | `string` | Yes | Câu truy vấn, NFC normalized, stripped |
| `relevant_article_ids` | `list[int]` | Yes | Sorted, unique IDs $\in [0, 59635]$ |

### 3.2. QA Benchmark Schema (`data/benchmarks/qa_{train,val,test}.parquet`)

| Cột | Parquet Type | Nullable | Mô tả & Ràng buộc |
| :--- | :--- | :--- | :--- |
| `qa_id` | `int64` | No | Định danh duy nhất toàn cục ($29,145$ unique) |
| `doc_id` | `int64` | No | ID tài liệu nguồn |
| `content_hash` | `string` | No | 32 hex MD5 hash của `context` (Group key) |
| `question` | `string` | No | Câu hỏi đánh giá, NFC normalized |
| `answer` | `string` | No | Câu trả lời tham chiếu chuẩn (Ground Truth) |
| `context` | `string` | No | Đoạn trích điều khoản chứa bằng chứng |
| `difficulty` | `string` | No | Tập giá trị: `easy`, `medium`, `hard` |
| `question_type` | `string` | No | Tập giá trị: `factual`, `application`, `interpretation`, `analytical` |

---

## 4. Thống Kê Phân Phối (Empirical Distributions)

### 4.1. Phân phối độ khó QA (Difficulty %)

| Mức độ | Toàn bộ (29,145) | Train (20,403) | Val (4,374) | Test (4,368) |
| :--- | :---: | :---: | :---: | :---: |
| **easy** | 33.36% | 33.36% | 33.42% | 33.31% |
| **medium** | 33.35% | 33.35% | 33.31% | 33.38% |
| **hard** | 33.29% | 33.29% | 33.26% | 33.31% |

### 4.2. Phân phối loại câu hỏi QA (Question Type %)

| Loại câu hỏi | Toàn bộ (29,145) | Train (20,403) | Val (4,374) | Test (4,368) |
| :--- | :---: | :---: | :---: | :---: |
| **factual** | 33.44% | 33.44% | 33.52% | 33.38% |
| **application** | 25.81% | 25.81% | 25.81% | 25.82% |
| **interpretation** | 22.10% | 22.10% | 22.04% | 22.12% |
| **analytical** | 18.65% | 18.64% | 18.63% | 18.68% |

---

## 5. Bằng Chứng Kiểm Định (Validation Proofs)

Các bộ dữ liệu đã vượt qua hai validator độc lập với exit code `0`:

1. **`retrieval_validation_report.txt`**:
   - `ALL ASSERTIONS PASSED (100%)`.
   - 100% ID nhãn tồn tại trong dải $[0, 59635]$.
   - Khắc phục triệt để 3 query bị gán trùng nhãn nội tại trong raw dataset.

2. **`qa_validation_report.txt`**:
   - `ALL ASSERTIONS PASSED (100%)`.
   - 0 giá trị null, 0 text rỗng.
   - 29,145 `qa_id` duy nhất.
   - Không có giao cắt ngữ cảnh giữa bất kỳ cặp tập nào:
     $$\text{Contexts}_{\text{train}} \cap \text{Contexts}_{\text{val}} = \emptyset$$
     $$\text{Contexts}_{\text{train}} \cap \text{Contexts}_{\text{test}} = \emptyset$$
     $$\text{Contexts}_{\text{val}} \cap \text{Contexts}_{\text{test}} = \emptyset$$