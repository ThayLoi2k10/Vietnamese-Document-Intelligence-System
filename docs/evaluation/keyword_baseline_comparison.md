# Báo Cáo Đối Chuẩn: TF-IDF vs. BM25 Tuned (Official Keyword Baseline)

Tài liệu này ghi lại chi tiết các phân tích so sánh đối chuẩn giữa mô hình TF-IDF và BM25 Tuned ($k_1=1.80, b=0.85$) trên 438 test queries thuộc `retrieval_test.json`.

## 1. Phân Phối Win / Tie / Loss

- **BM25 Wins:** 222 queries (50.68%)
- **Ties:** 165 queries (37.67%)
- **BM25 Losses:** 51 queries (11.64%)
- **Hard Failures ($R@10=0$ cả hai):** 99 queries (22.60%)

## 2. Hiệu Năng Phân Tầng (Single-label vs. Multi-label)

| Phân nhóm | Số queries | TF-IDF nDCG@10 | BM25 nDCG@10 | Delta nDCG@10 | TF-IDF R@10 | BM25 R@10 | Delta R@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single-label** | 329 | 0.3286 | 0.5183 | +0.1897 | 0.5289 | 0.7234 | +0.1945 |
| **Multi-label** | 109 | 0.2295 | 0.3910 | +0.1615 | 0.3250 | 0.4971 | +0.1721 |

## 3. Phân Tích Trường Hợp BM25 Losses (TF-IDF thắng)

Có tổng cộng **51** truy vấn mà TF-IDF đạt nDCG@10 cao hơn BM25. Dưới đây là 5 ví dụ tiêu biểu:

- **Query 3071:** "Trách nhiệm của người đại diện theo ủy quyền của chủ sở hữu, thành viên, cổ đông công ty là tổ chức như thế nào?"
  - Ground Truth: `[55452]`
  - TF-IDF nDCG@10: `1.0000` | BM25 nDCG@10: `0.5000` (Delta: `-0.5`)
  - TF-IDF R@10: `1.0000` | BM25 R@10: `1.0000`

- **Query 10357:** "Áp dụng pháp luật trong tương trợ tư pháp được thực hiện ra sao?"
  - Ground Truth: `[13778]`
  - TF-IDF nDCG@10: `0.5000` | BM25 nDCG@10: `0.4307` (Delta: `-0.0693`)
  - TF-IDF R@10: `1.0000` | BM25 R@10: `1.0000`

- **Query 3979:** "Nếu người lao động không nghỉ việc khi nhận con nuôi thì mức trợ cấp bảo hiểm xã hội một lần được hưởng sẽ là bao nhiêu tiền?"
  - Ground Truth: `[13370, 13371, 17679, 17684, 17686, 28189]`
  - TF-IDF nDCG@10: `0.3026` | BM25 nDCG@10: `0.1303` (Delta: `-0.1723`)
  - TF-IDF R@10: `0.1667` | BM25 R@10: `0.1667`

- **Query 9695:** "Từ chối kết quả trúng đấu giá tài sản của Bộ Quốc phòng như thế nào?"
  - Ground Truth: `[5632]`
  - TF-IDF nDCG@10: `0.6309` | BM25 nDCG@10: `0.5000` (Delta: `-0.1309`)
  - TF-IDF R@10: `1.0000` | BM25 R@10: `1.0000`

- **Query 13641:** "Thẩm quyền quyết định nhập, xuất hàng dự trữ quốc gia được quy định như thế nào?"
  - Ground Truth: `[37678]`
  - TF-IDF nDCG@10: `0.6309` | BM25 nDCG@10: `0.3333` (Delta: `-0.2976`)
  - TF-IDF R@10: `1.0000` | BM25 R@10: `1.0000`

## 4. Phân Tích Hard Failures (Giới Hạn Vật Lý Của Keyword Search)

Có **99** queries mà cả TF-IDF và BM25 đều không thể tìm thấy bất kỳ điều luật liên quan nào trong Top 10 ($R@10 = 0$).

Các đặc trưng cốt lõi của nhóm câu hỏi này:
1. **Hiện tượng diễn giải đồng nghĩa (Paraphrasing):** Người dùng dùng từ thông dụng đời thường thay cho thuật ngữ pháp lý.
2. **Truy vấn trừu tượng / suy luận (Abstract / Multi-hop):** Câu hỏi mô tả hành vi vi phạm mà không gọi tên chính xác điều luật.
3. **Khoảng cách từ vựng (Vocabulary Mismatch):** Tỷ lệ giao thoa ký tự với tài liệu đúng cực kỳ thấp.

5 ví dụ tiêu biểu về Hard Failures:

- **Query 591:** "Nhiệm vụ, quyền hạn của Ban Thanh tra nhân dân ở cơ quan, đơn vị?"
  - Ground Truth: `[5978]`
  - Multi-label: `False`

- **Query 4639:** "Tổ chức kinh doanh dịch vụ môi giới bất động sản có những nghĩa vụ gì?"
  - Ground Truth: `[885]`
  - Multi-label: `False`

- **Query 3275:** "Làm giám đốc công ty hợp danh khi là thành viên góp vốn công ty hợp danh hay không? Admin cho em hỏi thành viên góp vốn trong công ty hợp danh có thể làm giám đốc công ty không ạ?"
  - Ground Truth: `[55619]`
  - Multi-label: `False`

- **Query 7226:** "Bộ Tài nguyên và Môi trường được ban hành các văn bản quy phạm pháp luật nào?"
  - Ground Truth: `[42921]`
  - Multi-label: `False`

- **Query 2472:** "Những trường hợp thu hồi giấy chứng nhận đăng ký xe?"
  - Ground Truth: `[58097]`
  - Multi-label: `False`

## 5. Kết Luận Kỹ Thuật Chuyển Tiếp Sang Phase 6

- BM25 Tuned hoàn toàn vượt trội TF-IDF về mặt chuẩn hóa độ dài và kiểm soát bão hòa tần suất, xác lập mốc đối chuẩn chính thức: **nDCG@10 = 0.4866**, **Recall@10 = 66.71%**, **Latency = 3.26 ms**.
- Tuy nhiên, rào cản lớn nhất của phương pháp từ khóa là **99 queries thất bại hoàn toàn**, chiếm tỷ trọng đáng kể. Đây là giới hạn không thể khắc phục nếu chỉ dùng lexical matching.
- **Nhiệm vụ Phase 6 (Dense Retrieval):** Triển khai mô hình Bi-Encoder embedding ngữ nghĩa tiếng Việt nhằm thu hẹp khoảng cách từ vựng và xử lý triệt để nhóm Hard Failures này.
