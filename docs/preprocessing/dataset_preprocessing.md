# Tổng quan Chiến lược Tiền xử lý Dữ liệu (Vietnamese NLP Preprocessing Strategy)

## 1. Bối cảnh & Thách thức trong Xử lý Văn bản Pháp lý Tiếng Việt

Văn bản pháp lý tiếng Việt có các đặc tính kỹ thuật rất khắt khe:
- **Tính phân cấp chặt chẽ:** Văn bản được tổ chức theo cấu trúc hình cây: *Văn bản -> Phần -> Chương -> Mục -> Điều -> Khoản -> Điểm*.
- **Tính nguyên vẹn của số hiệu:** Các chuỗi như `Điều 1.`, `Khoản 2.`, `a)`, `Nghị định 24/2010/NĐ-CP` là tín hiệu phân đoạn ranh giới ngữ nghĩa (semantic boundary) cho chunking và retrieval. Xóa bỏ hoặc biến đổi dấu câu tùy tiện sẽ làm sụp đổ hoàn toàn khả năng định vị tài liệu.
- **Hiện tượng phân mảnh bảng mã (Encoding Fragmentation):** Dữ liệu thu thập từ nhiều nguồn công báo số hóa qua các thời kỳ thường bị trộn lẫn giữa Unicode dựng sẵn (NFC) và Unicode tổ hợp (NFD), đi kèm các ký tự điều khiển ẩn và rác định dạng HTML.

---

## 2. Các nguyên tắc tiền xử lý cốt lõi (Core Principles)

### 2.1. Chuẩn hóa Unicode NFC (Canonical Decomposition, followed by Canonical Composition)
- **Vấn đề:** Từ `"tiếng"` có thể được biểu diễn dưới dạng 1 ký tự dựng sẵn (`\u0074\u0069\u1ebf\u006e\u0067` - NFC) hoặc dạng tổ hợp phân rã (`\u0074\u0069\u0065\u0302\u0301\u006e\u0067` - NFD). Dù hiển thị trên màn hình giống nhau, nhưng đối với các tokenizer (Byte-Pair Encoding, WordPiece) hoặc các thuật toán so khớp chuỗi trong Python, chúng là các chuỗi ký tự khác nhau.
- **Quyết định kỹ thuật:** Toàn bộ text qua pipeline đều phải qua `unicodedata.normalize('NFC', text)`.
- **Mục tiêu đạt được:** NFC giúp đưa các biểu diễn Unicode tương đương về cùng một dạng canonical, tránh việc cùng một từ bị coi là các chuỗi khác nhau trong tokenization và matching. Điều này giúp loại bỏ hoàn toàn hiện tượng phân mảnh token giả tạo ở các phase Embedding (Phase 6) cũng như sai lệch chuỗi khi lập chỉ mục từ khóa (Phase 5).

### 2.2. Xử lý khoảng trắng ẩn & Khoảng trắng cấu trúc
- **Khử Non-Breaking Space (`\xa0` / NBSP):** Ánh xạ toàn bộ ký tự `\xa0` sinh ra từ định dạng web/WYSIWYG về khoảng trắng tiêu chuẩn (` `).
- **Chuẩn hóa ngắt dòng (Newline Harmonization):** Toàn bộ các định dạng ngắt dòng Windows (`\r\n`) và Mac cũ (`\r`) đều được quy chuẩn về định dạng Unix tiêu chuẩn (`\n`).
- **Chính sách bảo tồn khoảng trắng cấu trúc (Structural Whitespace Preservation):**
  - **Tuyệt đối không dùng** `re.sub(r'\s+', ' ', text)` trên toàn văn bản vì sẽ xóa mất cấu trúc dòng.
  - **Quy tắc:** Chỉ thu gọn khoảng trắng ngang (`[^\S\r\n\t]+ -> ' '`) trên từng dòng độc lập, bảo toàn nguyên vẹn `\t` và `\n` để phục vụ nhận diện thụt đầu dòng cấu trúc ở Phase 7 (Chunking).

### 2.3. Kiểm định chất lượng độc lập (Automated Validation Gate)
Mỗi luồng dữ liệu đều đi kèm một bộ kiểm định độc lập (`scripts/validation/`):
- Tự động kiểm tra tính liên tục của ID, độ phủ schema, tỷ lệ null/rỗng và tính toàn vẹn chuỗi.
- Tự động trả về mã thoát `sys.exit(0)` khi đạt hoặc `sys.exit(1)` khi có lỗi, phục vụ tích hợp CI/CD tự động ở Phase 23.

---

## 3. Tổng kết Cấu trúc Artifacts

```text
data/processed/
├── legal_retrieval/
│   ├── corpus_articles.parquet               # 59,636 articles được làm phẳng từ legal corpus (Staging)
│   ├── corpus_articles_preprocessed.parquet  # 59,636 articles sạch chuẩn NFC & metadata (Production)
│   └── validation_report.txt                 # Báo cáo kiểm định 11/11 PASS
├── legal_qa/
│   ├── legal_qa_preprocessed.parquet         # 29,145 cặp QA phẳng kèm phân loại 4 loại câu hỏi
│   └── validation_report.txt                 # Báo cáo kiểm định 6/6 PASS
└── legal_documents/
    ├── legal_documents_preprocessed.parquet  # 170,824 văn bản HTML bóc tách + metadata
    └── validation_report.txt                 # Báo cáo kiểm định 6/6 PASS