# Báo cáo Kỹ thuật: Tiền xử lý Vietnamese Legal QA Dataset (Stream 2)

## 1. Mục tiêu kỹ thuật
Chuyển đổi tập dữ liệu `vietnamese_legal_qa.parquet` từ cấu trúc mảng lồng (nested array of objects) thành bảng dữ liệu phẳng chuẩn hóa, làm sạch văn bản tiếng Việt và hợp nhất hệ thống nhãn phân loại câu hỏi (Taxonomy Harmonization) phục vụ xây dựng QA Benchmark (Phase 4), đánh giá RAG (Phase 10), và phân tích lỗi chuyên sâu (Phase 14 & Phase 29).

---

## 2. Kiến trúc Chuyển đổi Dữ liệu (`scripts/processing/preprocess_legal_qa.py`)
- **Nguồn đầu vào:** `data/raw/legal_qa/vietnamese_legal_qa.parquet` (9,715 dòng văn bản, mỗi dòng chứa mảng 3 câu hỏi trong cột `generated_qa_pairs`).
- **Phép biến đổi:** Phân rã mảng (Explode / Flattening) thành **29,145 cặp QA độc lập**.
- **Cấp phát định danh:**
  - `qa_id`: Khóa chính duy nhất, chạy liên tục từ $0 \to 29,144$.
  - `doc_id`: Khóa ngoại ánh xạ ngược về văn bản luật gốc ($0 \to 9,714$).
- **Chuẩn hóa văn bản:** Chuẩn hóa đồng bộ Unicode NFC, khử NBSP cho cả 3 trường text: `question`, `answer`, `article_content`.

---

## 3. Khắc phục Lỗi Thực nghiệm: Phân mảnh Nhãn Song ngữ (Bilingual Label Fragmentation)

### 3.1. Phát hiện lỗi qua Bộ kiểm định
Khi chạy validator lần đầu, hệ thống phát hiện lỗi phân mảnh nhãn do pipeline sinh dữ liệu trước đó không ràng buộc Enum:
- Cột `difficulty`: Bị trộn nhãn tiếng Anh (`easy`, `medium`, `hard`) với tiếng Việt (`dễ`, `trung bình`, `khó`).
- Cột `question_type`: Bị trộn nhãn tiếng Anh (`factual`, `interpretation`, `analytical`, `application`) với tiếng Việt (`thực tế`, `giải thích`, `ứng dụng`).

### 3.2. Bảng quy đổi chuẩn hóa (Taxonomy Harmonization Table)

| Trường dữ liệu | Giá trị thô | Giá trị sau chuẩn hóa | Số lượng bản ghi | Tỷ lệ (%) |
| :--- | :--- | :--- | :--- | :--- |
| **difficulty** | `easy`, `dễ` | **`easy`** | 9,724 | 33.36% |
| | `medium`, `trung bình` | **`medium`** | 9,719 | 33.35% |
| | `hard`, `khó` | **`hard`** | 9,702 | 33.29% |
| **question_type** | `factual`, `thực tế` | **`factual`** | 9,747 | 33.44% |
| | `application`, `ứng dụng` | **`application`** | 7,523 | 25.81% |
| | `interpretation`, `giải thích` | **`interpretation`** | 6,440 | 22.10% |
| | `analytical` | **`analytical`** | 5,435 | 18.65% |

### 3.3. Ý nghĩa thực tế cho Benchmark & RAG Evaluation (Phase 4, Phase 10 & Phase 29)
Bộ dữ liệu sau khi chuẩn hóa bao quát 4 dạng câu hỏi pháp lý theo mức độ suy luận:
1. **Factual:** Tra cứu dữ kiện trực tiếp có sẵn trong văn bản điều luật.
2. **Interpretation:** Diễn giải, làm rõ nghĩa từ ngữ hoặc phạm vi áp dụng của điều khoản.
3. **Application:** Áp dụng quy định luật để giải quyết một tình huống cụ thể.
4. **Analytical:** Phân tích, tổng hợp logic hoặc đối chiếu hệ quả pháp lý phức tạp.

Hệ thống phân loại này cung cấp ground truth để:
- Tạo các phân hoạch Train/Validation/Test theo Stratified Group Split dựa trên `content_hash`, bảo đảm không có context leakage giữa các tập ở Phase 4.
- Đánh giá khả năng bám sát ngữ cảnh (Faithfulness/Grounding) và độ chính xác câu trả lời ở Phase 10.
- Lập ma trận chẩn đoán lỗi ở Phase 14 và Phase 29: phân tách rõ lỗi do retrieval không lấy được context hay do LLM suy luận sai theo từng dạng câu hỏi.

---

## 4. Kết quả Kiểm định (Validation Results)
- **Script kiểm định:** `scripts/validation/validate_legal_qa.py`.
- **Báo cáo:** `data/processed/legal_qa/validation_report.txt`.
- **Chỉ số nghiệm thu (6/6 Checks Passed):**
  1. **Số lượng bản ghi:** Khớp chính xác 29,145 dòng.
  2. **Tính toàn vẹn khóa:** `qa_id` duy nhất và liên tục 100% trong dải $[0, 29144]$.
  3. **Tính đầy đủ văn bản:** 0 câu hỏi rỗng, 0 câu trả lời rỗng, 0 context rỗng.
  4. **Chuẩn hóa Taxonomy:** 100% giá trị cột `difficulty` thuộc `{'easy', 'medium', 'hard'}`.
  5. **Chuẩn hóa Phân loại:** 100% giá trị cột `question_type` thuộc `{'factual', 'application', 'interpretation', 'analytical'}`.
  6. **Cấu trúc dữ liệu:** Schema bảng phẳng chuẩn hóa, kiểu dữ liệu tương thích hoàn toàn cho các downstream pipeline.