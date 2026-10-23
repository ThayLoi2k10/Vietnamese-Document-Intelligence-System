# Phase 12 — Báo Cáo Thực Nghiệm Tái Xếp Hạng (Reranking Experimental Report)

**Dự án:** Hệ thống Trí tuệ Tài liệu Pháp luật Việt Nam (Vietnamese Legal Document Intelligence System)  
**Tập dữ liệu:** Đóng băng 438 câu hỏi kiểm thử (`data/benchmarks/retrieval_test.json`)  
**Corpus:** 59,636 điều luật (`corpus_articles_preprocessed.parquet`, ID space $[0, 59635]$)  
**Mô hình Reranker:** `BAAI/bge-reranker-base` (Cross-Encoder, XLM-RoBERTa backbone, 278M params, FP16 CUDA)  
**Bộ sinh ứng viên Tầng 1:** Hybrid Weighted RRF ($k=5, w_{\text{dense}}=0.75, w_{\text{bm25}}=0.25$) từ Phase 11  

---

## 1. Tóm Tắt Điều Hành & Phát Hiện Trọng Tâm (Executive Summary)

Phase 12 đánh giá thực nghiệm việc tích hợp mô hình **Cross-Encoder Reranker** (`BAAI/bge-reranker-base`) trên tập ứng viên được sinh ra từ bộ truy xuất kết hợp **Hybrid Retrieval** (Weighted RRF).

### Các phát hiện thực nghiệm cốt lõi:
1. **Phân tích Đối chuẩn Tổng thể:**
   - Trong Tầng 1 (First-stage retrieval), **Hybrid Weighted RRF** đạt đỉnh cao hiệu năng với **nDCG@10 = 0.6448**, **MRR@10 = 0.6366**, **Recall@100 = 0.9540** và độ trễ cực thấp **4.07 ms/query**.
   - Việc áp dụng mô hình Cross-Encoder zero-shot `BAAI/bge-reranker-base` trên nhóm ứng viên Top-10 đạt **nDCG@10 = 0.6096**, và trên nhóm Top-20 đạt **nDCG@10 = 0.5830**, với độ trễ tăng vọt lên **579.9 ms** (Top-10) và **1,155.7 ms** (Top-20).
2. **Quy luật Đánh đổi Độ sâu Ứng viên (Candidate Pool Scaling Trade-off):**
   - Thực nghiệm đo lường scaling trên tập chuẩn 100 queries cho thấy:
     - Candidate Pool Top-10: nDCG@10 = 0.6826 | Latency = 579.9 ms | QPS = 1.72
     - Candidate Pool Top-20: nDCG@10 = 0.6358 | Latency = 1,155.7 ms | QPS = 0.87
     - Candidate Pool Top-50: nDCG@10 = 0.6026 | Latency = 2,883.1 ms | QPS = 0.35
   - **Quy luật nghịch đảo độ sâu (Pool Depth Inversion):** Khi mở rộng candidate pool từ 10 lên 20 và 50, chất lượng xếp hạng giảm dần do mô hình zero-shot bị nhiễu bởi các "distractor" mang tính ngữ nghĩa chung chung.
3. **Phân tích Win/Tie/Loss:**
   - Mô hình Reranker ghi nhận **87 queries thắng (19.9%)** và **81 queries cải thiện MRR@10 (18.5%)**, nâng nhiều điều luật từ vị trí 8–15 lên Top-3.
   - Tuy nhiên, có **155 queries bị tụt hạng (35.4%)**, giải thích nguyên nhân sụt giảm metric trung bình vĩ mô (macro-average).
4. **Kết luận Kiến trúc Sản xuất:**
   - **Hybrid Retrieval (Weighted RRF)** tiếp tục được chọn làm **Production Retriever** chính thức nhờ sự tối ưu vượt trội về độ chính xác (nDCG@10 = 0.6448) và độ trễ thời gian thực (4.07 ms).
   - Reranker Cross-Encoder cần được fine-tune trực tiếp trên dữ liệu pháp lý Việt Nam với Hard Negatives chuyên ngành trước khi triển khai online.

---

## 2. Bảng So Sánh Đa Hệ Thống (Cross-System Benchmark Table)

Bảng đối chuẩn toàn diện 5 cấu hình trên 438 câu hỏi frozen test:

| Hệ thống / Cấu hình | nDCG@10 | MRR@10 | Recall@5 | Recall@10 | Recall@20 | Recall@100 | Precision@5 | Zero-hit@10 | Latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BM25 Tuned** ($k_1=1.80, b=0.85$) | 0.4866 | 0.4610 | 0.5644 | 0.6671 | 0.7468 | 0.8727 | 0.1370 | 31.96% | **3.26 ms** |
| **Dense Bi-Encoder** (BGE-M3) | 0.6277 | 0.6146 | 0.7018 | 0.7969 | 0.8555 | 0.9441 | 0.1685 | 13.93% | 17.50 ms |
| **Hybrid Weighted RRF** (Phase 11) | **0.6448** | **0.6366** | **0.7273** | **0.8013** | **0.8675** | **0.9540** | **0.1763** | **12.33%** | **4.07 ms** |
| **Hybrid + Reranker (Top-10 Pool)** | 0.6096 | 0.5885 | 0.6856 | **0.8013** | **0.8675** | **0.9540** | 0.1639 | **12.33%** | 579.87 ms |
| **Hybrid + Reranker (Top-20 Pool)** | 0.5830 | 0.5656 | 0.6396 | 0.7659 | **0.8675** | **0.9540** | 0.1539 | 15.53% | 1,155.66 ms |

---

## 3. Thực Nghiệm Đánh Đổi Kích Thước Candidate Pool (Candidate Pool Size Trade-offs)

Để kiểm chứng giả thuyết về độ sâu ứng viên đầu vào (Candidate Pool Depth) theo yêu cầu đặc tả Phase 12, thực nghiệm so sánh Top-10, Top-20, và Top-50 được thực hiện trên **tập calibration subset gồm đúng 100 queries cố định trích xuất từ frozen test set (`retrieval_test.json`), không phải toàn bộ 438 queries** (nhằm tối ưu hóa tài nguyên tính toán GPU trên cấu hình phần cứng thử nghiệm).

```
[Candidate Pool Depth vs Quality & Latency on 100-query Calibration Subset]
Pool Size 10: [nDCG@10: 0.6826] ----------> Latency: 580 ms (QPS: 1.72)
Pool Size 20: [nDCG@10: 0.6358] --------------------> Latency: 1,156 ms (QPS: 0.87)
Pool Size 50: [nDCG@10: 0.6026] ---------------------------------------------> Latency: 2,883 ms (QPS: 0.35)
```

| Candidate Pool Size | nDCG@10 | MRR@10 | Recall@5 | Recall@10 | Precision@5 | Latency/query (ms) | Throughput (QPS) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Top-10 Candidates** | **0.6826** | **0.6662** | **0.7303** | **0.8753** | **0.1660** | **579.87 ms** | **1.72** |
| **Top-20 Candidates** | 0.6358 | 0.6296 | 0.7070 | 0.7903 | 0.1600 | 1,155.66 ms | 0.87 |
| **Top-50 Candidates** | 0.6026 | 0.5997 | 0.6720 | 0.7420 | 0.1540 | 2,883.06 ms | 0.35 |

> [!NOTE]
> Bảng trên thể hiện xu hướng tỷ lệ nghịch giữa độ sâu ứng viên và chất lượng xếp hạng (nDCG@10 giảm dần từ 0.6826 xuống 0.6026 khi mở rộng pool từ 10 lên 50) trên tập chuẩn 100 câu hỏi. Đối với toàn bộ benchmark 438 câu hỏi, hai cấu hình chính Top-10 và Top-20 đã được đánh giá đầy đủ ở Mục 2.

### Nhận xét chuyên sâu:
- **Hiện tượng suy giảm khi mở rộng Pool (Negative Dilution):** Khi đưa 50 ứng viên vào Cross-Encoder, các điều luật ở vị trí 21–50 thường chứa nhiều từ khóa định nghĩa chung (ví dụ: các điều về nguyên tắc, thẩm quyền chung, giải thích từ ngữ) có điểm cross-attention bề mặt cao hơn điều luật cụ thể, dẫn đến việc đẩy lùi các điều luật ground-truth xuống dưới Top-10.
- **Chi phí suy luận:** Độ trễ tăng tuyến tính nghiêm ngặt với số cặp câu hỏi – điều luật ($O(M)$). Top-50 tiêu tốn gần 3 giây cho mỗi truy vấn trên GPU, không khả thi cho môi trường tương tác người dùng thời gian thực nếu không có cụm tăng tốc chuyên biệt.

---

## 4. Tác Động Hệ Thống RAG Hạ Nguồn (Downstream RAG Impact)

Trong kiến trúc RAG, ngữ cảnh được truyền vào LLM thường bị giới hạn ở **Top-5 điều luật** để bảo vệ context window và hạn chế hallucination.

> [!IMPORTANT]
> **Phân biệt phương pháp luận:** Các chỉ số `Context Precision@5` và `Context Recall@5` trong bảng dưới đây là **Retrieval Context Cutoff Metrics** (tương đương với $P@5$ và $R@5$ ở tầng truy xuất văn bản điều luật đầu vào của context window). Chúng **hoàn toàn khác biệt** với các chỉ số **Generative Citation Precision / Recall** của Phase 10 (vốn đo lường trích dẫn thực tế được sinh ra trong câu trả lời tự nhiên của LLM thông qua parser trích xuất văn bản sinh ra).

| Hệ thống Truy xuất Context | Context Precision@5 (Retrieval P@5) | Context Recall@5 (Retrieval R@5) | Mật độ Thông tin Đúng (Signal-to-Noise) |
|---|:---:|:---:|:---:|
| **BM25 Tuned** | 0.1370 | 0.5644 | Thấp (nhiều điều luật trùng từ khóa giả) |
| **Dense BGE-M3** | 0.1685 | 0.7018 | Khá cao (nắm bắt ngữ nghĩa tốt) |
| **Hybrid Weighted RRF** | **0.1763** | **0.7273** | **Tối ưu nhất (Bao phủ 72.73% căn cứ pháp lý)** |
| **Hybrid + Reranker (Top-10)** | 0.1639 | 0.6856 | Tốt trên câu hỏi diễn giải, giảm trên câu hỏi thủ tục |
| **Hybrid + Reranker (Top-20)** | 0.1539 | 0.6396 | Nhiễu tăng do distractor lọt vào Top-5 |

**Kết luận RAG:** Đối với bài toán RAG văn bản pháp luật, tầng truy xuất Tầng 1 Hybrid đã cung cấp context có độ phủ và độ tinh khiết cao nhất cho LLM generation. Việc thêm Cross-Encoder zero-shot chưa qua fine-tune làm giảm nhẹ Context Recall@5 từ 72.73% xuống 68.56%.

---

## 5. Phân Tích Chi Tiết Từng Câu Hỏi (Query-Level Win/Tie/Loss Analysis)

Phân tích đối đầu trực tiếp giữa **Hybrid + Reranker (Top-20)** và **Hybrid Baseline** trên 438 câu hỏi:

```
[nDCG@10 Win/Tie/Loss Distribution]
Wins (Reranker tốt hơn):    87 queries (19.86%)  ████
Ties (Bằng điểm):          196 queries (44.75%)  █████████
Losses (Hybrid tốt hơn):   155 queries (35.39%)  ███████
```

### 5.1. Các ca Thắng tiêu biểu (Strong Wins):
Reranker phát huy ưu thế vượt trội ở các câu hỏi có **tính lập luận ngữ nghĩa phức tạp hoặc diễn giải gián tiếp**:
1. **Query 4639:** *"Tổ chức kinh doanh dịch vụ môi giới bất động sản có những nghĩa vụ gì?"*
   - Hybrid nDCG@10: 0.6309 $\to$ **Reranked: 1.0000** (+0.3691)
   - Điều luật ground-truth được đưa chính xác lên Rank 1.
2. **Query 1912:** *"Thời hạn kháng nghị bản án, quyết định hình sự theo thủ tục giám đốc thẩm được quy định ra sao?"*
   - Hybrid nDCG@10: 0.0000 $\to$ **Reranked: 0.5000** (+0.5000)
   - Điều luật căn cứ bị trôi ra ngoài Top-10 trong Hybrid đã được Reranker phát hiện và kéo lên Rank 3.
3. **Query 7304:** *"Khai thác nước biển để làm muối thì có cần xin phép cơ quan nhà nước hay không?"*
   - Hybrid nDCG@10: 0.6309 $\to$ **Reranked: 1.0000** (+0.3691)
   - Nắm bắt chính xác mối liên hệ điều kiện - miễn trừ trong luật tài nguyên nước.

### 5.2. Các ca Thua tiêu biểu (Strong Losses) & Nguyên nhân:
1. **Query 591:** *"Nhiệm vụ, quyền hạn của Ban Thanh tra nhân dân ở cơ quan, đơn vị?"*
   - Hybrid nDCG@10: 0.3155 $\to$ **Reranked: 0.0000** (-0.3155)
   - *Nguyên nhân:* Cross-Encoder ưu tiên các điều luật về "Quyền hạn của Thủ trưởng cơ quan" và "Nhiệm vụ của Thanh tra Nhà nước" (chứa nhiều token trùng lặp hơn) thay vì điều luật cụ thể về Ban Thanh tra nhân dân.
2. **Query 3071:** *"Trách nhiệm của người đại diện theo ủy quyền của chủ sở hữu, thành viên, cổ đông công ty là tổ chức như thế nào?"*
   - Hybrid nDCG@10: 1.0000 $\to$ **Reranked: 0.5000** (-0.5000)
   - Điều luật đúng bị đẩy từ Rank 1 xuống Rank 2 do một điều khoản về "Đại diện theo pháp luật" chiếm vị trí đầu.

---

## 6. Phân Tích Lỗi Sâu Sắc (In-Depth Error Analysis)

Tại sao một mô hình Cross-Encoder mạnh mẽ như `bge-reranker-base` lại gặp khó khăn trước Hybrid Baseline trong môi trường pháp điển Việt Nam?

1. **Khoảng cách miền (Domain Shift) & Biểu diễn ngôn ngữ pháp lý:**
   - Văn bản pháp luật Việt Nam có cấu trúc phân tầng cao (`Điều`, `Khoản`, `Điểm`, `Căn cứ`) với tính logic hình thức chặt chẽ.
   - `bge-reranker-base` được huấn luyện chủ yếu trên văn bản web tổng quát (Wikipedia, QA forum, web search). Nó có xu hướng đánh giá cao các đoạn văn chứa nhiều từ khóa định nghĩa rộng thay vì các điều khoản quy định xử phạt / thủ tục cụ thể.
2. **Vấn đề "Bẫy Từ Khóa Định Nghĩa" (Definition Article Bias):**
   - Trong Bộ luật, Điều 2 hoặc Điều 3 thường là "Giải thích từ ngữ". Các điều này tập trung hầu hết thuật ngữ của câu hỏi nên được mô hình chấm điểm cao giả tạo, chiếm chỗ của các điều luật quy định nội dung thực chất.
3. **Độ dài ngữ cảnh và Padding:**
   - Các điều luật có độ dài trung bình 184 từ, nhưng phương sai lớn (một số điều dài trên 1,000 từ). Việc giới hạn `max_length=256` hoặc `512` có thể cắt ngang các khoản quy định ở phần sau của điều luật.

---

## 7. Khuyến Nghị Kỹ Thuật & Lộ Trình (Engineering Recommendations)

1. **Khuyến nghị Triển khai Hiện tại:**
   - **Giữ nguyên Hybrid Weighted RRF làm kiến trúc tìm kiếm sản xuất mặc định.** Không kích hoạt Reranker zero-shot online để tránh làm tăng độ trễ thêm 1,150 ms và làm giảm 6.18% nDCG@10.
2. **Định hướng Tinh chỉnh Reranker (Domain Adaptation Roadmap):**
   - Muốn Cross-Encoder vượt qua Hybrid, bắt buộc phải thực hiện **Domain Fine-Tuning**:
     - Thu thập bộ ba `(Query, Positive Article, Hard Negative Article)` từ tập `retrieval_train.json`.
     - Sử dụng chính các điều luật xếp hạng từ 2 đến 10 của BM25 và Dense làm Hard Negatives.
     - Huấn luyện Cross-Encoder với hàm mất mát InfoNCE hoặc MarginRankingLoss.
3. **Cấu hình Tối ưu nếu bắt buộc sử dụng Reranker:**
   - Nếu áp dụng trong pipeline QA, chỉ nên rerank với **Candidate Pool Top-10** (nDCG@10 = 0.6096, độ trễ 580 ms), tuyệt đối không mở rộng lên Top-50 trong điều kiện suy luận CPU/GPU phổ thông.

---

## 8. Danh Mục Artifacts Của Phase 12

Các tệp kết quả và mã nguồn hoàn chỉnh đã được lưu trữ và kiểm chứng:
1. Đặc tả kỹ thuật: [docs/phase12_reranking_specification.md](file:///D:/PROJECT/PROJECT%202/docs/phase12_reranking_specification.md)
2. Báo cáo thực nghiệm: [docs/evaluation/reranking_report.md](file:///D:/PROJECT/PROJECT%202/docs/evaluation/reranking_report.md)
3. Metrics kiểm thử chính thức: [results/benchmarks/reranked_test_metrics.json](file:///D:/PROJECT/PROJECT%202/results/benchmarks/reranked_test_metrics.json)
4. Dự đoán chi tiết trên 438 queries: [results/benchmarks/reranked_test_predictions.json](file:///D:/PROJECT/PROJECT%202/results/benchmarks/reranked_test_predictions.json)
5. Báo cáo đánh đổi Candidate Pool scaling: [results/benchmarks/reranker_candidate_pool_scaling.json](file:///D:/PROJECT/PROJECT%202/results/benchmarks/reranker_candidate_pool_scaling.json)
6. Bảng so sánh đa hệ thống: [results/benchmarks/reranker_vs_baselines_comparison.json](file:///D:/PROJECT/PROJECT%202/results/benchmarks/reranker_vs_baselines_comparison.json)
7. Thư viện Reranking:
   - Module độc lập: [src/retrieval/reranking/cross_encoder.py](file:///D:/PROJECT/PROJECT%202/src/retrieval/reranking/cross_encoder.py)
   - Adapter RAG Phase 9: [src/rag/reranking/adapters/cross_encoder_reranker.py](file:///D:/PROJECT/PROJECT%202/src/rag/reranking/adapters/cross_encoder_reranker.py)
8. Bộ unit test: [tests/test_reranker.py](file:///D:/PROJECT/PROJECT%202/tests/test_reranker.py) (5/5 tests passed).
