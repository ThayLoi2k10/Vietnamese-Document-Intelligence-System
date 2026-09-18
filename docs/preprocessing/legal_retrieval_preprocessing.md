# Báo cáo Kỹ thuật: Tiền xử lý Vietnamese Legal Retrieval Corpus (Stream 1)

## 1. Mục tiêu kỹ thuật
Chuyển đổi tập dữ liệu thô `legal_corpus.json` (dung lượng ~111 MB chứa các văn bản luật dạng cây phân cấp) thành bảng dữ liệu phẳng (tabular Parquet) ở cấp độ **Điều (Article-level)**, sẵn sàng làm retrieval corpus ở cấp độ Article, phục vụ Keyword Search (Phase 5), Dense Retrieval (Phase 6) và RAG Chunking (Phase 7).

---

## 2. Kiến trúc Pipeline 2 Chặng (Two-stage Ingestion & Preprocessing)

### Chặng 1: Ingestion & Làm phẳng cấu trúc (`scripts/processing/build_corpus_article.py`)
- **Nguồn đầu vào:** `data/raw/legal_retrieval/legal_corpus.json`.
- **Logic thực hiện:** Bóc tách mảng lồng `articles` bên trong từng văn bản luật (`law_id`).
- **Sản phẩm trung gian (Staging Artifact):** `data/processed/legal_retrieval/corpus_articles.parquet`.
  - Quy mô: **59,636 articles**.
  - Cấp phát khóa chính toàn cục: `global_article_id` chạy liên tục từ $0 \to 59,635$.
  - Lưu giữ nguyên bản 100% chuỗi text thô chưa chỉnh sửa để làm mốc đối chứng (Ground Truth Preservation).

### Chặng 2: Tiền xử lý chuyên sâu (`scripts/processing/preprocess_corpus_articles.py`)
- Áp dụng chuẩn hóa Unicode NFC bằng `unicodedata.normalize('NFC', text)`.
- Khử sạch ký tự `\xa0` (NBSP) và chuyển đổi ngắt dòng Windows/Mac cũ CRLF/CR $\to$ `\n`.
- Thu gọn khoảng trắng ngang trên từng dòng, bảo toàn nguyên vẹn cấu trúc thụt đầu dòng.
- Tính toán các trường phái sinh: `char_count_preprocessed`, `word_count_preprocessed`, `is_empty_preprocessed`.
- **Sản phẩm hoàn thiện (Production Artifact):** `data/processed/legal_retrieval/corpus_articles_preprocessed.parquet`.
  - Schema chuẩn 10 cột: `[global_article_id, doc_id, law_id, aid, text_raw, text, char_count_preprocessed, word_count_preprocessed, is_empty_preprocessed, created_at]`.

---

## 3. Các phát hiện thực nghiệm cốt lõi (Empirical Findings)

### 3.1. Ký tự TAB (`\t`) và bài toán Structural Preservation
- **Khảo sát:** Phân tích dữ liệu thực nghiệm phát hiện có tới **380,496 ký tự `\t`** nằm rải rác trong 59,636 articles.
- **Ý nghĩa cấu trúc:** Trong văn bản quy phạm pháp luật Việt Nam, ký tự `\t` được dùng để phân tách số thứ tự các Điểm, Khoản (ví dụ: `1.\tPhạm vi điều chỉnh...`, `a)\tNội dung...`).
- **Kết quả nghiệm thu:** Số lượng ký tự `\t` trước tiền xử lý là **380,496**; sau tiền xử lý vẫn bảo toàn chính xác **380,496** (độ lệch bằng 0).

### 3.2. Hiện tượng Điều rỗng (Empty Articles)
- Kiểm tra logic nội dung phát hiện **8 articles hoàn toàn không có nội dung chữ** (chuỗi trống hoặc chỉ chứa khoảng trắng).
- **Quyết định xử lý:** Không xóa bỏ 8 bản ghi này để bảo toàn tính liên tục của `global_article_id` ($0 \to 59,635$). Thay vào đó, gán cờ boolean `is_empty_preprocessed = True` để các downstream pipeline (Inverted Indexing ở Phase 5, Dense Indexing ở Phase 6) tự động lọc bỏ khi nạp tài liệu.

---

## 4. Kết quả Kiểm định (Validation Results)
- **Script kiểm định:** `scripts/validation/validate_preprocessed_corpus.py`.
- **Báo cáo:** `data/processed/legal_retrieval/validation_report.txt`.
- **Chỉ số nghiệm thu (11/11 Validation Sections PASS):**
  1. **Row Count Check:** Đúng chính xác 59,636 / 59,636 articles (độ phủ 100%).
  2. **Schema & Columns Check:** Đúng cấu trúc 10 cột chuẩn, không thiếu cột nào.
  3. **Global Article ID Continuity Check:** Khóa chính liên tục từ $0 \to 59,635$, không trùng lặp, không khuyết thiếu.
  4. **Primary Key / Foreign Key Integrity Check:** Toàn bộ `doc_id`, `law_id`, `aid` hợp lệ và không chứa giá trị null.
  5. **Unicode Normalization Check (NFC):** 0 bản ghi lỗi NFD, 100% đạt chuẩn canonical NFC.
  6. **Artifact Cleanliness Check (NBSP & CRLF):** 0 bản ghi chứa `\xa0`, 0 bản ghi chứa ký tự ngắt dòng `\r`.
  7. **Structural Preservation Check (`\t` & `\n`):** Bảo toàn nguyên vẹn 380,496 ký tự `\t` và toàn bộ cấu trúc xuống dòng `\n`.
  8. **Text Length & Word Count Check:** Các trường `char_count_preprocessed` và `word_count_preprocessed` nhất quán với nội dung `text`.
  9. **Empty Flag Alignment Check:** Khớp chính xác 8 bản ghi rỗng với điều kiện `is_empty_preprocessed == True`.
  10. **Null / NaN Value Check:** 0 giá trị null bất thường trong các trường dữ liệu bắt buộc.
  11. **Data Types & Parquet Metadata Check:** Kiểu dữ liệu tương thích hoàn toàn (int64, string, boolean, timestamp) với chuẩn Apache Arrow/Parquet.