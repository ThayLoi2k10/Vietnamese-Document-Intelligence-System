# Báo cáo Kỹ thuật: Tiền xử lý Vietnamese Legal Documents (Stream 3)

## 1. Mục tiêu kỹ thuật
Bóc tách toàn văn nội dung sạch từ 170,824 văn bản HTML phân mảnh trên 12 file partition, loại bỏ mã boilerplate định dạng web và tích hợp các trường metadata cần thiết phục vụ Document Intelligence, Metadata Filtering và đồ thị quan hệ văn bản. Dữ liệu `relationships` chưa được đưa vào artifact này mà được lưu trữ độc lập cho pipeline phân tích quan hệ riêng.

---

## 2. Kiến trúc Streaming Pipeline (`scripts/processing/preprocess_legal_documents.py`)
Do khối lượng dữ liệu lớn (170,824 chuỗi HTML thô nặng hàng trăm MB), pipeline được thiết kế theo cơ chế **Partition-by-Partition Streaming** để tối ưu tài nguyên:
1. Nạp trước bảng Metadata (171,556 dòng) vào bộ nhớ để lập chỉ mục tra cứu nhanh theo khóa `id`.
2. Đọc lần lượt từng partition trong số 12 file Parquet của `data/raw/legal_documents/content/`.
3. Bóc tách text thuần từ HTML bằng `BeautifulSoup` và làm sạch rác định dạng.
4. Thực hiện `LEFT JOIN` giữa Content và Metadata theo khóa liên kết `id`.
5. Ghi dữ liệu tuần tự ra file đích `legal_documents_preprocessed.parquet` bằng `pyarrow.parquet.ParquetWriter` với chuẩn nén `snappy`.

> **Lưu ý về bộ nhớ:** Content được xử lý tuần tự theo từng partition, trong khi toàn bộ metadata được nạp trước vào RAM để tối ưu tốc độ join. Pipeline này streaming theo content partition, không phải streaming hoàn toàn 100% dữ liệu.

---

## 3. Khắc phục các Thách thức Kỹ thuật Thực tế (Edge Cases Solved)

### 3.1. Xử lý Tràn bộ nhớ & Boilerplate HTML
- **Vấn đề:** Nếu dùng `BeautifulSoup.get_text()` thông thường, toàn bộ mã CSS nằm trong thẻ `<style>` và thuộc tính trong thẻ `<meta>` sẽ bị coi là text, làm ô nhiễm dữ liệu tìm kiếm.
- **Giải pháp:** Sử dụng phương thức `decompose()` triệt tiêu hoàn toàn các nút cây DOM không chứa nội dung: `["script", "style", "head", "meta", "title"]` trước khi trích xuất text.

### 3.2. Xử lý Xung đột Ký tự Toán học & Regex HTML (Math Symbols vs HTML Tags)
- **Vấn đề phát hiện:** Khi kiểm thử với regex bắt thẻ HTML, hệ thống phát hiện 81 văn bản vi phạm:
  - Một số văn bản kỹ thuật chứa công thức so sánh: ví dụ Thông tư 06/2018/TT-BTTTT có chuỗi: `39 dBm < p < 43 dBm` (trong đó $p$ là công suất). Regex thông thường bắt nhầm `< p <` thành thẻ `<p>`.
  - Một số văn bản chứa dấu so sánh chưa unescape: `30%<fe≤40%< p="">`.
- **Giải pháp:** Tinh chỉnh bộ lọc chuyên biệt:
  - Khử chính xác các thẻ lỗi unescaped dạng `<\s*/?[a-zA-Z0-9_≤≥%]+(?:\s+[^>]*)?>`.
  - Regex kiểm định trong validator được thắt chặt bằng word boundary: `<\s*/?(?:div|span|html|body|script|style)\b|(?<!\w)</?p[>\s]`, loại bỏ hoàn toàn việc bắt nhầm ký hiệu toán học.

### 3.3. Hiện tượng 12,223 Văn bản Rỗng Nội dung
- **Khảo sát:** Có **12,223 văn bản** (~7.16%) có `char_count == 0` sau khi bóc tách HTML.
- **Bản chất nghiệp vụ:** Các văn bản này là các Quyết định ban hành Bảng giá đất hoặc Định mức kỹ thuật. Toàn bộ nội dung thực tế nằm trong các phụ lục PDF/Excel đính kèm, trang HTML gốc chỉ chứa thông tin trích yếu.
- **Quyết định kỹ thuật:**
  - Tuyệt đối không xóa bỏ các dòng này nhằm bảo toàn đầy đủ các nút văn bản cho đồ thị quan hệ (`relationships`).
  - Bổ sung cờ boolean `is_empty`:
    - `is_empty = False`: **158,601 văn bản** (92.84%) $\to$ Đưa vào Indexing phục vụ Retrieval.
    - `is_empty = True`: **12,223 văn bản** (7.16%) $\to$ Giữ lại metadata, bỏ qua khâu index nội dung văn bản.

---

## 4. Kết quả Kiểm định (Validation Results)
- **Script kiểm định:** `scripts/validation/validate_legal_documents.py`
- **Báo cáo:** `data/processed/legal_documents/validation_report.txt`
- **Chỉ số nghiệm thu (6/6 Checks Passed):**
  1. **Độ phủ dữ liệu:** 170,824 / 170,824 bản ghi (100.0%).
  2. **Tính toàn vẹn khóa:** 100% `id` duy nhất tuyệt đối, 0 khóa null.
  3. **Độ phủ Metadata:** `title` đạt 99.99%, `loai_van_ban` đạt 100.00%.
  4. **Độ sạch HTML:** 0 tàn dư thẻ HTML hợp lệ trong toàn bộ các mẫu kiểm tra.
  5. **Tính nhất quán của cờ boolean:** `is_empty == True` khớp 100% với điều kiện `char_count == 0`.
  6. **Cấu trúc cột và kiểu dữ liệu:** Khớp 100% schema chuẩn phục vụ downstream pipelines.