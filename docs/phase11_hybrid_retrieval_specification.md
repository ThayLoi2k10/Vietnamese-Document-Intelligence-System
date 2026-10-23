# Phase 11 — Đặc tả Kỹ thuật Truy xuất Hỗn hợp (Hybrid Retrieval Specification)

## 1. Bối cảnh và Mục tiêu (Context & Objective)

Trong các pha trước:
- **Phase 5 (Lexical Baselines):** BM25 tuned ($k_1=1.80, b=0.85$) đạt **nDCG@10 = 0.4866**, **Recall@10 = 0.6671**, **Recall@100 = 0.8727**. BM25 vượt trội khi truy vấn chứa từ khóa kỹ thuật, số hiệu luật chính xác (ví dụ: *Điều 15*, *Nghị định 167/2018/NĐ-CP*), nhưng gặp thất bại nặng nề khi người dùng diễn đạt bằng ngôn ngữ tự nhiên hoặc từ đồng nghĩa (99/438 câu test có Recall@10 = 0).
- **Phase 6 (Dense Bi-Encoder Baseline):** `BAAI/bge-m3` đạt **nDCG@10 = 0.6277**, **Recall@10 = 0.7969**, **Recall@100 = 0.9441**. Dense retrieval xử lý xuất sắc sự tương đồng ngữ nghĩa và diễn giải tương đương, nhưng đôi khi bỏ qua các từ khóa định danh cụ thể.

**Mục tiêu Phase 11:** Xây dựng và đánh giá thực nghiệm mô hình **Hybrid Retrieval** cấp độ điều luật (Article-level) bằng cách kết hợp sức mạnh bổ trợ của **BM25** và **BGE-M3 Dense Retrieval** trên toàn bộ corpus **59,636 điều luật** (`corpus_articles_preprocessed.parquet`).

---

## 2. Phạm vi & Ràng buộc Bất biến (Scope & Constraints)

1. **Cấp độ truy xuất:** Duy nhất cấp độ điều luật (**Article-level**). Không tráo đổi sang chunk-level BM25/Qdrant trong thực nghiệm cốt lõi này.
2. **Không gian định danh:** Bảo toàn 100% không gian `global_article_id` $\in [0, 59635]$.
3. **Tập dữ liệu đối chuẩn:**
   - **Tập huấn luyện/tinh chỉnh (Train split):** `data/benchmarks/retrieval_train.json` (1,752 queries) dùng để tinh chỉnh siêu tham số fusion ($k_{\text{rrf}}, w_{\text{dense}}, w_{\text{bm25}}$).
   - **Tập kiểm thử đóng băng (Frozen test split):** `data/benchmarks/retrieval_test.json` (438 queries) TUYỆT ĐỐI KHÔNG dùng để chọn tham số hay quyết định cấu hình.
4. **Tái sử dụng chỉ mục:**
   - Chỉ mục BM25: `data/indices/bm25_index.pkl` ($k_1=1.80, b=0.85$).
   - Chỉ mục Dense: `data/indices/dense_bge_m3/index.faiss` + mô hình `BAAI/bge-m3`.

---

## 3. Các Phương pháp Dung hợp (Fusion Algorithms)

### 3.1. Reciprocal Rank Fusion (RRF)
RRF kết hợp danh sách xếp hạng của hai mô hình dựa trên vị trí thứ hạng (rank), không phụ thuộc vào biên độ điểm số (score scale):

$$\text{RRF\_Score}(d) = \frac{w_{\text{dense}}}{k_{\text{rrf}} + \text{rank}_{\text{dense}}(d)} + \frac{w_{\text{bm25}}}{k_{\text{rrf}} + \text{rank}_{\text{bm25}}(d)}$$

Trong đó:
- $\text{rank}_m(d) \in [1, K]$ là thứ hạng 1-indexed của tài liệu $d$ trong top $K$ kết quả trả về của retriever $m$ ($K=100$).
- Nếu tài liệu $d$ không xuất hiện trong top $K$ của retriever $m$, thành phần tương ứng bằng 0.
- $k_{\text{rrf}} \ge 1$ là hằng số làm mịn (smoothing constant).
- **Standard RRF:** $w_{\text{dense}} = 0.5, w_{\text{bm25}} = 0.5$ (tương đương trọng số bằng nhau).
- **Weighted RRF:** $w_{\text{dense}} + w_{\text{bm25}} = 1.0$ với $w_{\text{dense}} \in [0.5, 0.9]$.

### 3.2. Linear Score Combination (Convex Score Fusion)
Chuẩn hóa điểm số Min-Max độc lập trên từng danh sách truy xuất của mỗi query:

$$S_{\text{norm}}(d) = \frac{S(d) - \min_j S(d_j)}{\max_j S(d_j) - \min_j S(d_j) + \epsilon}$$

$$\text{Linear\_Score}(d) = \alpha \cdot S_{\text{norm, dense}}(d) + (1 - \alpha) \cdot S_{\text{norm, bm25}}(d)$$

---

## 4. Giao thức Thực nghiệm và Tiêu chí Đánh giá (Protocol & Metrics)

1. **Thước đo cốt lõi:**
   - `P@k`, `R@k` với $k \in \{1, 5, 10, 20, 50, 100\}$
   - `MRR@10`
   - `MAP@10`, `MAP@100`
   - `nDCG@10`, `nDCG@100`
   - `Zero-hit rate`: Tỷ lệ câu hỏi không tìm thấy tài liệu liên quan tại Top-10 và Top-100.
   - `Latency (ms/query)`
2. **Quy tắc chọn mô hình (Selection Criterion):**
   - Lựa chọn cấu hình dựa trên **Train nDCG@10** trên 1,752 câu `retrieval_train.json`.
3. **Phân tích Win/Tie/Loss:**
   - So sánh từng câu truy vấn: Hybrid cải thiện (Win), giữ nguyên (Tie) hay suy giảm (Loss) nDCG@10 so với Dense baseline và BM25 baseline.
