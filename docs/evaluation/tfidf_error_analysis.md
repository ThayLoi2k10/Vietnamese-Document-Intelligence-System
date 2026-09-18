# Phân Tích Lỗi Thực Nghiệm: TF-IDF Baseline

Tài liệu này ghi lại chi tiết các failure modes của mô hình `TF-IDF` trên 438 queries thuộc `retrieval_test.json`.

---

## 1. Bức Tranh Tổng Thể Tại Top-10

- **Tổng số truy vấn test:** 438
- **Thành công trọn vẹn ($R@10 = 1.0$):** 183 queries (41.78%)
- **Thành công một phần ($0 < R@10 < 1.0$):** 54 queries (12.33%)
- **Thất bại hoàn toàn ($R@10 = 0.0$):** 201 queries (45.89%)

---

## 2. Khoảng Cách Hiệu Năng Giữa Single-label và Multi-label

| Nhóm truy vấn | Số lượng | MRR@10 | Recall@10 | MAP@10 | nDCG@10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Single-label** | 329 | 0.2664 | 0.5289 | 0.2664 | 0.3286 |
| **Multi-label** | 109 | 0.2810 | 0.3250 | 0.1495 | 0.2295 |

> **Nhận xét:** Trên các truy vấn Multi-label (chiếm 24.89%), Recall@10 sụt giảm đáng kể so với Single-label. Điều này phản ánh rõ hạn chế của TF-IDF: khi câu hỏi đòi hỏi tổng hợp từ nhiều điều luật khác nhau, cơ chế so khớp từ khóa thuần túy chỉ có thể bắt được điều luật trùng từ khóa nhiều nhất, bỏ sót các điều luật liên đới.

---

## 3. Phân Loại Nguyên Nhân Thất Bại (Error Taxonomy)

Trong **201** queries thất bại hoàn toàn tại Top-10 ($R@10 = 0.0$), các trường hợp phân rã thành hai nhóm nguyên nhân cốt lõi:

### A. Lexical Chasm / Vocabulary Mismatch
- **Bản chất:** Người dùng hỏi bằng ngôn ngữ tự nhiên, từ đồng nghĩa hoặc câu hỏi suy diễn thực tế, trong khi điều luật quy định bằng thuật ngữ pháp lý chuyên ngành. Trùng lặp từ vựng thực chất giữa query và ground-truth gần như bằng 0 (hoặc chỉ trùng các hư từ/từ chức năng).
- **Ví dụ tiêu biểu:**
  * **Query ID 10495:** *"Tiêu chuẩn Trưởng khoa/phòng của báo, tạp chí và trung tâm truyền thông của Bộ Y tế được quy định như thế nào?"*
    - Ground-truth ID: `58135`
    - Top 1 retrieved ID: `40775` (Score: 0.4176)
    - Hiện tượng: Trùng các từ chung chung (`và`, `trung`, `khoa`, `phòng`, `trưởng`), nhưng các từ định danh cơ quan chuyên ngành không đủ sức kéo đúng điều luật mục tiêu.
  * **Query ID 6690:** *"Học mầm non ở nông thôn phải đóng bao nhiêu tiền học phí mỗi tháng?"*
    - Ground-truth ID: `59635`
    - Top 1 retrieved ID: `6967` (Score: 0.3739)
    - Hiện tượng: Ngôn ngữ giao tiếp đời thường (*"ở nông thôn"*, *"đóng bao nhiêu tiền học phí"*) lệch hoàn toàn với văn phong quy định trong văn bản (*"mức thu học phí đối với cơ sở giáo dục mầm non công lập tại vùng nông thôn"*).
  * **Query ID 6920:** *"Có bắt buộc phải về quê xin Giấy xác nhận tình trạng hôn nhân sau khi nhập hộ khẩu ở địa chỉ mới không?"*
    - Ground-truth ID: `5469`
    - Top 1 retrieved ID: `56351` (Score: 0.4246)
    - Hiện tượng: Từ ngữ người hỏi (*"về quê"*, *"nhập hộ khẩu"*) không xuất hiện trong quy định đăng ký hộ tịch và tình trạng hôn nhân.

### B. Lexical Competition & Ranking Bias (Thiếu Term Saturation)
- **Bản chất:** Query có trùng từ khóa với văn bản đúng, nhưng các điều luật khác chứa các từ khóa đó lặp lại nhiều lần hoặc chứa nhiều từ khóa phụ của query nhận điểm TF tích lũy quá cao, đẩy điều luật đúng ra ngoài Top 10.
- **Ví dụ tiêu biểu:**
  * **Query ID 591:** *"Nhiệm vụ, quyền hạn của Ban Thanh tra nhân dân ở cơ quan, đơn vị?"*
    - Ground-truth ID: `5978`
    - Top 1 retrieved ID: `11902` (Score: 0.5906)
    - Hiện tượng: Các từ `thanh tra`, `nhân dân`, `nhiệm vụ`, `quyền hạn` xuất hiện dày đặc trong nhiều văn bản luật khác nhau, dẫn tới việc các điều luật dài có tổng điểm dot product cao hơn điều luật cụ thể cần tìm.
  * **Query ID 8068:** *"Xây dựng, ban hành kế hoạch thanh tra như thế nào?"*
    - Ground-truth ID: `51357`
    - Top 1 retrieved ID: `39422` (Score: 0.4622)
    - Hiện tượng: Cụm từ `kế hoạch thanh tra` xuất hiện ở hàng chục điều luật quy định về thẩm quyền, đẩy quy trình xây dựng cụ thể tụt xuống hạng thấp.

---

## 4. Kết Luận & Định Hướng Cải Thiện

1. **BM25 (Baseline 1B):** Giải quyết trực tiếp nhóm lỗi **B (Lexical Competition)** thông qua cơ chế bão hòa tần suất từ (*Term Saturation* kiểm soát bởi $k_1$) và chuẩn hóa độ dài tài liệu tường minh (*Document Length Normalization* qua tham số $b$).
2. **Dense Retrieval (Baseline 2 — Phase 6):** Giải quyết nhóm lỗi **A (Lexical Chasm)** bằng không gian vector ngữ nghĩa liên tục (Bi-Encoder embedding), ánh xạ các cách diễn đạt ngôn ngữ đời thường về gần với thuật ngữ pháp lý.