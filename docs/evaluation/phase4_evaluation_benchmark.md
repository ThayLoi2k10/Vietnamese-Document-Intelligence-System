# Phase 4 — Xây Dựng & Kiểm Chuẩn Tập Dữ Liệu Đánh Giá (Evaluation Benchmark)

Tài liệu này ghi lại chi tiết quy trình thiết kế, tiền xử lý và đặc tả kỹ thuật của tập dữ liệu kiểm chuẩn phục vụ bài toán Information Retrieval (IR) và RAG trên văn bản pháp luật Việt Nam.

---

## 1. Tổng Quan Corpus Pháp Luật (Document Collection)

- **Tệp dữ liệu:** `data/processed/legal_retrieval/corpus_articles_preprocessed.parquet`
- **Quy mô:** 59,636 articles được chuẩn hóa từ hệ thống văn bản quy phạm pháp luật Việt Nam.
- **Định danh thống nhất:** `global_article_id` $\in [0, 59635]$.
- **Cấu trúc trường thông tin (Production Schema 10 cột):**
  - `global_article_id` (int64): Khóa chính định danh duy nhất của từng article trong toàn bộ hệ thống.
  - `doc_id` (int64): Định danh số của văn bản luật gốc trong raw corpus.
  - `law_id` (string): Số hiệu văn bản quy phạm pháp luật (ví dụ: `"14/2022/TT-NHNN"`).
  - `aid` (int64): Số thứ tự article cục bộ bên trong văn bản luật.
  - `text_raw` (string): Nội dung văn bản gốc chưa qua xử lý (lưu trữ đối chứng).
  - `text` (string): Nội dung article đã được chuẩn hóa Unicode NFC, khử NBSP và chuẩn hóa ngắt dòng.
  - `char_count_preprocessed` (int64): Số lượng ký tự sau tiền xử lý.
  - `word_count_preprocessed` (int64): Số lượng từ sau tiền xử lý.
  - `is_empty_preprocessed` (bool): Cờ đánh dấu article rỗng nội dung chữ (8 articles rỗng được gán `True`).
  - `created_at` (timestamp): Thời điểm thực thi pipeline.

---

## 2. Thiết Kế Phân Vùng Benchmark (Train / Test Split Protocol)

Để bảo đảm tính khách quan và ngăn chặn hiện tượng rò rỉ dữ liệu (Data/Label Leakage), tập câu hỏi truy vấn được phân chia nghiêm ngặt:

| Tập dữ liệu | Đường dẫn tệp | Số lượng Queries | Số cặp Relevance Ground-Truth | Tỷ lệ Multi-label | Mục đích sử dụng |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Train Set** | `data/benchmarks/retrieval_train.json` | 1,752 | 2,348 | 24.83% | Tinh chỉnh siêu tham số (Hyperparameter Tuning), huấn luyện Retriever/Reranker |
| **Test Set (Hold-out)** | `data/benchmarks/retrieval_test.json` | 438 | 588 | 24.89% | Đóng băng tuyệt đối; chỉ dùng để đánh giá đối chuẩn cuối cùng |

### Nguyên tắc kỹ thuật:
1. **Zero Leakage:** Tập `retrieval_test.json` được khóa cứng hoàn toàn trong toàn bộ chu kỳ phát triển. Mọi quyết định lựa chọn kiến trúc, tinh chỉnh trọng số chỉ được tối ưu dựa trên tập Train.
2. **Đại diện truy vấn thực tế:** Tập câu hỏi bao gồm cả truy vấn đơn nhãn (1 câu hỏi $\rightarrow$ 1 article, chiếm ~75.1%) và truy vấn đa nhãn (1 câu hỏi $\rightarrow$ từ 2 đến nhiều articles liên quan, chiếm ~24.9%).

---

## 3. Cấu Trúc Bản Ghi Benchmark (JSON Schema)

Mỗi truy vấn trong tập benchmark tuân thủ cấu trúc dữ liệu sau:

```json
{
  "query_id": 1042,
  "query_text": "thời hiệu khởi kiện yêu cầu bồi thường thiệt hại ngoài hợp đồng",
  "relevant_article_ids": [7464, 14721]
}

query_id (int): Định danh duy nhất của truy vấn.

query_text (str): Nội dung câu truy vấn bằng tiếng Việt tự nhiên, được chuẩn hóa Unicode NFC và whitespace.

relevant_article_ids (List[int]): Danh sách article IDs thuộc ground truth, được chuẩn hóa thành sorted unique IDs.

Ghi chú: Thuộc tính is_multi_label được suy ra trong quá trình xây dựng benchmark từ số lượng phần tử của relevant_article_ids và được sử dụng làm biến phân tầng khi chia Train/Test; thuộc tính này không được lưu như một trường riêng trong artifact JSON.