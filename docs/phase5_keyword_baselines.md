# Phase 5 — Baseline 1: Keyword Search (TF-IDF & BM25)

Tài liệu này ghi lại toàn bộ cơ sở lý thuyết, kiến trúc hệ thống, quá trình tinh chỉnh tham số và kết quả đối chuẩn thực nghiệm của các thuật toán tìm kiếm từ khóa truyền thống (Lexical Baselines).

---

## 1. Thiết Kế Kiến Trúc Nền Tảng (System Contracts)

Toàn bộ các thuật toán truy xuất trong dự án được chuẩn hóa thông qua Interface trừu tượng:
* **Giao thức:** `BaseRetriever` (`src/retrieval/base.py`)
  * `fit(corpus)`: Nạp và lập chỉ mục văn bản.
  * `search(query_text, top_k)`: Tìm kiếm Top-$k$ cho một truy vấn đơn lẻ.
  * `batch_search(queries, top_k)`: Tìm kiếm tối ưu hàng loạt.
  * `save(path)` / `load(path)`: Lưu và nạp chỉ mục từ ổ đĩa.
* **Định dạng kết quả thống nhất:** `RetrievalResult(article_id: int, score: float, rank: int)`
* **Bộ đo lường độc lập:** `src/evaluation/retrieval_metrics.py`
  * Đo lường: Precision@k, Recall@k ($k \in \{1, 5, 10, 20, 50, 100\}$).
  * Đo chất lượng xếp hạng: MRR@10, MAP@10, MAP@100, nDCG@10, nDCG@100.
  * Đo hiệu năng hệ thống: Latency (ms/query), Throughput (QPS).

---

## 2. Triển Khai & Đánh Giá Baseline 1A: TF-IDF

### A. Thiết lập kỹ thuật
* **Mã nguồn:** `src/retrieval/tfidf_retriever.py`
* **Cấu hình:** Unigram, $L_2$-normalized, `sublinear_tf=True` ($1 + \log(\text{TF})$).
* **Quy mô chỉ mục:** 13,216 terms; ma trận thưa $59,636 \times 13,216$ (độ thưa: 99.27%); kích thước lưu trữ: 48.31 MB (`data/indices/tfidf_index.joblib`).

### B. Kết quả thực nghiệm trên Test Set (438 queries)
* **Latency:** $20.22\text{ ms/query}$ | **Throughput:** $49.26\text{ QPS}$
* **Chất lượng xếp hạng:**
  * $P@1 = 0.1644$, $P@5 = 0.0895$, $P@10 = 0.0575$
  * $R@1 = 0.1403$, $R@5 = 0.3760$, $R@10 = 0.4781$, $R@100 = 0.8261$
  * $\text{MRR@10} = 0.2700$, $\text{MAP@10} = 0.2373$, $\text{nDCG@10} = 0.3039$

### C. Phân tích lỗi (Error Analysis)
Trong 438 queries, các trường hợp thất bại tại Top-10 ($R@10 = 0.0$) phân rã thành 2 nhóm bản chất:
* **Lexical Chasm (Lệch từ khóa):** Query người dùng dùng từ ngữ đời thường, không giao thoa đủ từ vựng pháp lý với điều luật cần tìm.
* **Lexical Competition / Ranking Failure Candidates:** Query có trùng từ khóa với ground-truth, nhưng các điều luật khác nhận điểm tích lũy TF-IDF cao hơn khiến ground-truth bị đẩy ra ngoài Top-10.
* *Định hướng:* Cơ chế bão hòa tần suất từ (Term Frequency Saturation) và chuẩn hóa độ dài văn bản của BM25 là hướng xử lý trực tiếp cho nhóm cạnh tranh thứ hạng này.

---

## 3. Triển Khai & Đánh Giá Baseline 1B: BM25 (Okapi BM25)

### A. Thiết lập kỹ thuật
* **Mã nguồn:** `src/retrieval/bm25_retriever.py`
* **Kiến trúc:** Inverted Index kết hợp tính toán vector hóa NumPy.
* **Corpus thống kê:** Từ vựng: 13,284 terms; Tổng bản ghi postings: 6,136,944; Độ dài tài liệu trung bình ($\text{avgdl}$): $305.90$ từ.
* **Kích thước chỉ mục:** 48.25 MB (`data/indices/bm25_index.pkl`).

### B. Tinh chỉnh siêu tham số trên Train Set
Quét lưới 30 cấu hình ($k_1 \in [0.6, 2.1]$, $b \in [0.40, 0.85]$) trên 1,752 queries tập Train:
* **Cấu hình tối ưu được chọn:** $k_1^* = 1.80$, $b^* = 0.85$.
* *Giải thích cơ chế:* Cấu hình $b = 0.85$ cho kết quả tốt nhất trong phạm vi grid search đã thử. Một giả thuyết là mức chuẩn hóa độ dài cao hơn giúp giảm ảnh hưởng của các điều luật dài; $k_1 = 1.80$ đồng thời cho phép mức bão hòa TF phù hợp hơn với các thuật ngữ chuyên ngành lặp lại. Đây là diễn giải cơ chế, không phải kết luận causal độc lập từ experiment.

---

## 4. Bảng Tổng Hợp Đối Chuẩn Thực Nghiệm (Hold-out Test Set - 438 Queries)

| Metric | TF-IDF (Baseline 1A) | BM25 Default ($k_1=1.5, b=0.75$) | BM25 Tuned ($k_1=1.8, b=0.85$) | Chênh lệch Tuned vs. TF-IDF |
| :--- | :---: | :---: | :---: | :---: |
| **Latency (ms/query)** | 20.22 ms | 6.57 ms | **3.26 ms** | **Nhanh hơn 6.2 lần** |
| **Throughput (QPS)** | 49.26 | 148.68 | **301.09** | **+511.1%** |
| **P@1** | 0.1644 | 0.3196 | **0.3288** | **+100.0%** |
| **P@5** | 0.0895 | 0.1352 | **0.1370** | +53.07% |
| **P@10** | 0.0575 | 0.0811 | **0.0817** | +42.09% |
| **R@1** | 0.1403 | 0.2771 | **0.2846** | **+102.8%** |
| **R@5** | 0.3760 | 0.5562 | **0.5644** | +50.11% |
| **R@10** | 0.4781 | 0.6601 | **0.6671** | **+39.53%** |
| **R@20** | 0.6097 | 0.7396 | **0.7468** | +22.49% |
| **R@50** | 0.7428 | 0.8151 | **0.8231** | +10.81% |
| **R@100** | 0.8261 | 0.8537 | **0.8727** | +5.64% |
| **MRR@10** | 0.2700 | 0.4539 | **0.4610** | **+70.74%** |
| **MAP@10** | 0.2373 | 0.4049 | **0.4114** | **+73.37%** |
| **MAP@100** | 0.2543 | 0.4159 | **0.4227** | +66.22% |
| **nDCG@10** | 0.3039 | 0.4799 | **0.4866** | **+60.12%** |
| **nDCG@100** | 0.3827 | 0.5253 | **0.5339** | +39.51% |

---

## 5. Phân Tích Đóng Góp Kỹ Thuật (Win-Loss & Failure Analysis)

### A. Phân phối Win / Tie / Loss (theo nDCG@10, $\epsilon = 10^{-4}$)
* **BM25 Wins:** 222 queries (**50.68%**) — BM25 cải thiện trực tiếp thứ hạng so với TF-IDF.
* **Ties:** 165 queries (**37.67%**) — Hai mô hình có hiệu năng tương đương.
* **BM25 Losses:** 51 queries (**11.64%**) — TF-IDF xếp hạng tốt hơn ở một số điều luật đặc thù.

### B. Hiệu năng phân tầng (Single-label vs. Multi-label)
* **Single-label (329 queries):** Recall@10 tăng từ $52.89\% \to \mathbf{72.34\%}$; nDCG@10 tăng từ $0.3286 \to \mathbf{0.5183}$.
* **Multi-label (109 queries):** Recall@10 tăng từ $32.50\% \to \mathbf{49.71\%}$; nDCG@10 tăng từ $0.2295 \to \mathbf{0.3910}$.

### C. Giới hạn vật lý của Keyword Search (Hard Failures)
* Xác định chính xác **99 queries (22.60%)** cả TF-IDF và BM25 đều hoàn toàn không tìm thấy tài liệu đúng trong Top 10 ($R@10 = 0.0$).
* *Nguyên nhân cốt lõi:* Khoảng cách từ vựng (Vocabulary Mismatch) khi người dùng diễn đạt bằng ngôn ngữ đời thường thay vì thuật ngữ luật. Đây là rào cản vật lý của sparse matching, định hình bài toán cần giải quyết cho Dense Retrieval (Phase 6).

---

## 6. Official Baseline Chốt Cho Toàn Dự Án

* **Mô hình chuẩn:** **BM25 Tuned ($k_1=1.80, b=0.85$)**
* **Chỉ số tham chiếu chính thức:**
  * $\text{nDCG@10} = \mathbf{0.4866}$
  * $\text{Recall@10} = \mathbf{66.71\%}$
  * $\text{MRR@10} = \mathbf{0.4610}$
  * $\text{Recall@100} = \mathbf{87.27\%}$
  * $\text{Latency} = \mathbf{3.26\text{ ms/query}}$