# Information Retrieval Fundamentals — Phase 5

Tài liệu này ghi lại nền tảng lý thuyết Information Retrieval (IR) được sử dụng để thiết kế, triển khai và đánh giá các hệ thống **lexical retrieval** trong Project 2 — Vietnamese Legal Document Intelligence.

Mục tiêu của tài liệu là làm rõ bài toán ranking, cơ chế của TF-IDF và BM25, các tham số ảnh hưởng đến ranking, cùng các metric dùng để đánh giá retrieval. Các nội dung lý thuyết được đối chiếu với implementation thực tế trong `src/retrieval/` và `src/evaluation/`.

---

## 1. Bài toán Information Retrieval

### 1.1. Retrieval là gì?

Information Retrieval (IR) là bài toán tìm kiếm các tài liệu có liên quan đến một truy vấn của người dùng trong một corpus lớn.

Trong Project 2:

- **Corpus:** 59,636 article-level legal documents.
- **Query:** câu hỏi/truy vấn tìm kiếm từ bộ retrieval benchmark.
- **Document:** một điều luật/article được định danh bởi `global_article_id`.
- **Relevance:** được xác định bởi ground-truth article IDs trong benchmark.

Bài toán có thể biểu diễn:

$$q \rightarrow \{d_1, d_2, \dots, d_N\}$$

Trong đó hệ thống cần tính điểm relevance cho từng document và trả về danh sách được **xếp hạng giảm dần theo score**:

$$d_{(1)}, d_{(2)}, \dots, d_{(k)}$$

Mục tiêu không chỉ là tìm được document đúng, mà quan trọng hơn là đưa document liên quan lên vị trí càng cao càng tốt.

---

## 2. Lexical Retrieval

Lexical retrieval đánh giá mức độ liên quan chủ yếu dựa trên sự xuất hiện và tần suất của các term trong query và document.

Hai baseline đầu tiên của Phase 5 là:

1. **TF-IDF**
2. **BM25**

Đây là các sparse retrieval methods. Chúng hoạt động dựa trên lexical overlap thay vì học trực tiếp semantic similarity giữa query và document.

### 2.1. Ưu điểm

- Dễ giải thích.
- Không cần training neural model.
- Có thể xây dựng nhanh và reproducible.
- Hiệu quả tốt khi query sử dụng từ vựng tương đồng với document.
- Là baseline quan trọng trước khi đánh giá dense retrieval.

### 2.2. Hạn chế

Lexical retrieval gặp khó khăn khi:

- Query và document sử dụng các cách diễn đạt khác nhau.
- Người dùng dùng ngôn ngữ đời thường thay vì thuật ngữ pháp lý.
- Synonym hoặc paraphrase làm giảm lexical overlap.
- Các document dài chứa nhiều term phổ biến gây cạnh tranh ranking.

Do đó Phase 5 cung cấp baseline để xác định giới hạn của lexical matching trước khi chuyển sang dense retrieval.

---

## 3. Inverted Index

### 3.1. Ý tưởng

Thay vì duyệt toàn bộ corpus cho từng query, hệ thống xây dựng một **inverted index**.

Inverted index ánh xạ:

$$\text{term} \rightarrow \{\text{documents containing term}\}$$

Ví dụ:

```text
"đất"          → document 10, 25, 91, ...
"thuế"         → document 3, 25, 77, ...
"doanh nghiệp" → document 15, 42, ...
```

Khi query chứa một term, hệ thống chỉ cần truy cập postings list tương ứng thay vì quét toàn bộ corpus theo cách brute-force.

### 3.2. Vai trò trong Project 2

BM25 được triển khai bằng inverted index tự xây dựng trong:

```text
src/retrieval/bm25_retriever.py
```

Index thực tế có:
- **Vocabulary:** 13,284 terms
- **Tổng postings:** 6,136,944
- **Average document length:** 305.90 tokens

Các thống kê này mô tả cấu trúc lexical corpus được sử dụng cho BM25.

---

## 4. Term Frequency (TF)

TF biểu diễn mức độ xuất hiện của một term trong document.

Một dạng đơn giản:

$$\text{TF}(t, d) = f(t, d)$$

Trong đó:
- $t$: term.
- $d$: document.
- $f(t, d)$: số lần term xuất hiện trong document.

TF cao thường cho thấy term có mức độ liên quan lớn hơn với document.

Tuy nhiên, sử dụng TF tuyến tính có vấn đề: nếu một term xuất hiện rất nhiều lần, score có thể tiếp tục tăng mạnh dù những lần xuất hiện bổ sung không còn mang nhiều thông tin mới.

Đây là lý do BM25 sử dụng term-frequency saturation.

---

## 5. Inverse Document Frequency (IDF)

IDF đo mức độ phân biệt của một term trong toàn corpus.

Một term xuất hiện trong rất nhiều document có giá trị phân biệt thấp.
Ngược lại, một term xuất hiện trong ít document có giá trị phân biệt cao.

Một dạng IDF phổ biến:

$$\text{IDF}(t) = \log \frac{N}{\text{df}(t)}$$

Trong đó:
- $N$: tổng số document.
- $\text{df}(t)$: số document chứa term $t$.

Ý tưởng cốt lõi:

$$\text{df}(t) \uparrow \implies \text{IDF}(t) \downarrow$$

Do đó các từ phổ biến đóng góp ít hơn vào ranking.

---

## 6. TF-IDF

TF-IDF kết hợp hai thành phần:

$$\text{TFIDF}(t, d) = \text{TF}(t, d) \cdot \text{IDF}(t)$$

Ý tưởng:
- TF cao $\rightarrow$ term xuất hiện nhiều trong document.
- IDF cao $\rightarrow$ term hiếm trong corpus.
- Cả hai cao $\rightarrow$ term có khả năng đóng góp mạnh vào relevance score.

### 6.1. Implementation trong Project 2

TF-IDF được triển khai trong:

```text
src/retrieval/tfidf_retriever.py
```

Các đặc điểm chính:
- Unigram.
- `sublinear_tf=True`.
- $L_2$ vector normalization.
- Similarity được tính bằng dot product giữa vector query và vector document sau normalization.

Với sublinear TF:

$$\text{TF}_{\text{sublinear}} = 1 + \log(\text{TF}) \quad (\text{với } \text{TF} > 0)$$

Điều này làm giảm ảnh hưởng của việc một term lặp lại quá nhiều lần.

Sau $L_2$ normalization:

$$\Vert{}\mathbf{v}\Vert{}_2 = 1$$

Similarity giữa query và document có thể tính bằng:

$$\text{sim}(q, d) = \mathbf{v}_q^T \mathbf{v}_d$$

và tương đương với cosine similarity.

---

## 7. BM25

BM25 là một lexical ranking function được thiết kế để xử lý tốt hơn một số hạn chế của TF-IDF, đặc biệt là:
- Term-frequency saturation.
- Document-length normalization.

Dạng BM25 được sử dụng trong implementation của Project 2:

$$\text{score}(d, q) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d)(k_1 + 1)}{f(t, d) + k_1 \left(1 - b + b \frac{\vert{}d\vert{}}{\text{avgdl}}\right)}$$

Trong đó:
- $f(t, d)$: term frequency của $t$ trong document $d$.
- $\vert{}d\vert{}$: độ dài document.
- $\text{avgdl}$: độ dài document trung bình của corpus.
- $k_1$: điều khiển mức độ saturation của term frequency.
- $b$: điều khiển mức độ document-length normalization.

Implementation hiện tại sử dụng công thức Robertson-Spärck Jones IDF:

$$\text{IDF}(t) = \log \left(1 + \frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5}\right)$$

Đây là công thức IDF thực tế trong:

```text
src/retrieval/bm25_retriever.py
```

---

## 8. Hai cơ chế quan trọng của BM25

### 8.1. Term Frequency Saturation — $k_1$

Trong TF-IDF, việc tăng TF có thể tiếp tục làm score tăng. BM25 thay đổi hành vi này bằng một hàm có tính chất saturation.

Khi $f(t, d)$ tăng:

$$\frac{f(t, d)(k_1 + 1)}{f(t, d) + k_1(\dots)}$$

tăng nhưng tốc độ tăng giảm dần.

Do đó:
- Lần xuất hiện đầu tiên của term có giá trị lớn;
- Các lần lặp lại tiếp theo vẫn có giá trị;
- Nhưng đóng góp biên giảm dần.

#### Vai trò của $k_1$

$k_1$ điều khiển mức độ saturation của TF. Giá trị lớn hơn cho phép TF tiếp tục có ảnh hưởng trong phạm vi rộng hơn trước khi saturation mạnh.

Trong Phase 5:

```text
BM25 Default:
k1 = 1.5
b  = 0.75
```

Sau grid search trên training benchmark:

```text
Best configuration in tested grid:
k1 = 1.8
b  = 0.85
```

Đây là kết quả thực nghiệm trong 30 cấu hình đã thử, không phải một giá trị lý thuyết mặc định.

### 8.2. Document Length Normalization — $b$

BM25 điều chỉnh score theo độ dài document thông qua:

$$1 - b + b \frac{\vert{}d\vert{}}{\text{avgdl}}$$

Nếu:
- $\vert{}d\vert{} > \text{avgdl}$: document dài hơn mức trung bình và bị normalization mạnh hơn.
- $\vert{}d\vert{} < \text{avgdl}$: document ngắn hơn mức trung bình và chịu ít penalty hơn.

#### Vai trò của $b$

- $b = 0$: không thực hiện document-length normalization.
- $b = 1$: normalization đầy đủ theo độ dài document.
- Giá trị giữa 0 và 1 tạo ra mức normalization trung gian.

Trong Phase 5, grid search cho kết quả tốt nhất tại:

$$b = 0.85$$

trong phạm vi các giá trị được thử.

Một giả thuyết có thể giải thích kết quả là mức normalization cao hơn giúp giảm ảnh hưởng của các document dài. Tuy nhiên đây là diễn giải cơ chế, không phải kết luận causal độc lập từ experiment.

---

## 9. TF-IDF vs. BM25

| Đặc điểm | TF-IDF | BM25 |
| :--- | :--- | :--- |
| **Lexical matching** | Có | Có |
| **Term frequency** | Có | Có |
| **IDF** | Có | Có |
| **TF saturation** | Sublinear TF | Có |
| **Vector normalization** | $L_2$ normalization | Không áp dụng |
| **Document length normalization** | Không explicit | Explicit $b$ |
| **Inverted index** | Không bắt buộc | Có trong implementation |
| **Hyperparameters chính** | Vectorizer parameters | $k_1, b$ |
| **Semantic matching** | Không | Không |

Điểm quan trọng là BM25 không biến lexical retrieval thành semantic retrieval. Nó vẫn phụ thuộc vào lexical overlap giữa query và document.

---

## 10. Ranking trong Retrieval

Sau khi tính score:

$$\text{score}(q, d)$$

retriever sắp xếp document theo thứ tự giảm dần:

$$\text{score}(q, d_{(1)}) \ge \text{score}(q, d_{(2)}) \ge \dots$$

Sau đó trả về Top-$k$:

$$[d_{(1)}, d_{(2)}, \dots, d_{(k)}]$$

Trong Project 2, kết quả được chuẩn hóa thành:

```text
RetrievalResult(
    article_id,
    score,
    rank
)
```

với `article_id` tương ứng với `global_article_id` của article-level corpus.

---

## 11. Đánh giá Retrieval

Retrieval không nên chỉ đánh giá bằng một metric duy nhất. Project 2 sử dụng hai nhóm metric:

### 11.1. Retrieval coverage
- Precision@1, 5, 10, 20, 50, 100.
- Recall@1, 5, 10, 20, 50, 100.

### 11.2. Ranking quality
- MRR@10.
- MAP@10, MAP@100.
- nDCG@10, nDCG@100.

Ngoài chất lượng ranking, hệ thống còn được đo:
- **Latency** (ms/query).
- **Throughput** (QPS).

---

## 12. Precision@k

Precision@k đo tỷ lệ document trong Top-$k$ là relevant:

$$P@k = \frac{\vert{}\text{relevant retrieved documents in top } k\vert{}}{k}$$

Ví dụ, nếu Top-10 có 2 relevant documents:

$$P@10 = \frac{2}{10} = 0.2$$

Precision tập trung vào độ chính xác của những gì hệ thống trả về.

---

## 13. Recall@k

Recall@k đo tỷ lệ relevant documents được tìm thấy trong Top-$k$:

$$R@k = \frac{\vert{}\text{relevant retrieved documents in top } k\vert{}}{\vert{}\text{total relevant documents}\vert{}}$$

Ví dụ có 4 relevant documents và hệ thống tìm được 3 trong Top-10:

$$R@10 = \frac{3}{4} = 0.75$$

Recall đặc biệt quan trọng đối với legal retrieval vì một query có thể có nhiều article liên quan.

---

## 14. Mean Reciprocal Rank — MRR

Reciprocal Rank của một query được xác định bởi vị trí của relevant document đầu tiên:

$$\text{RR} = \frac{1}{\text{rank}_{\text{first relevant}}}$$

Nếu relevant document đầu tiên ở rank 1:

$$\text{RR} = 1$$

Nếu ở rank 5:

$$\text{RR} = 0.2$$

MRR là trung bình RR trên toàn bộ queries:

$$\text{MRR} = \frac{1}{\vert{}Q\vert{}} \sum_{q \in Q} \text{RR}(q)$$

Trong Project 2 sử dụng:

$$\text{MRR@10}$$

tức chỉ xét relevant document đầu tiên trong Top-10. MRR phù hợp khi vị trí của relevant result đầu tiên có ý nghĩa lớn.

---

## 15. Mean Average Precision — MAP

Average Precision (AP) xét chất lượng ranking của nhiều relevant documents trong một query.

Một dạng truncated AP@k được sử dụng trong implementation:

$$\text{AP@}k = \frac{\sum_{i=1}^{k} P@i \cdot \text{rel}_i}{\min(\vert{}\mathcal{R}\vert{}, k)}$$

Trong đó:
- $\text{rel}_i = 1$ nếu document tại rank $i$ relevant.
- $\text{rel}_i = 0$ nếu không relevant.
- $\vert{}\mathcal{R}\vert{}$ là số relevant documents của query.

MAP là trung bình AP trên toàn bộ queries:

$$\text{MAP@}k = \frac{1}{\vert{}Q\vert{}} \sum_{q \in Q} \text{AP@}k(q)$$

Implementation loại bỏ duplicate retrieved hits khi tính relevance contribution.

---

## 16. nDCG

nDCG (Normalized Discounted Cumulative Gain) đánh giá ranking bằng cách giảm trọng số của các kết quả nằm ở vị trí thấp hơn.

DCG@k:

$$\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$

Sau đó chuẩn hóa theo ranking lý tưởng:

$$\text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$$

Trong benchmark của Project 2, relevance được biểu diễn theo ground-truth article membership ($\text{rel}_i \in \{0, 1\}$).

Các metric được sử dụng:

$$\text{nDCG@10}, \quad \text{nDCG@100}$$

nDCG đặc biệt hữu ích cho bài toán ranking vì nó quan tâm đến vị trí của relevant documents chứ không chỉ việc chúng có xuất hiện hay không.

---

## 17. Latency và Throughput

Retrieval quality không phải yếu tố duy nhất.

### 17.1. Latency

Latency đo thời gian xử lý một query:

$$\text{Latency} = \frac{\text{total retrieval time}}{\text{number of queries}}$$

Đơn vị: ms/query. Latency thấp hơn thuận lợi hơn cho hệ thống interactive.

### 17.2. Throughput

Throughput biểu diễn số query xử lý được trong một đơn vị thời gian:

$$\text{QPS} = \frac{\text{number of queries}}{\text{total time}}$$

Đơn vị: queries/second.

Latency và QPS phải được xem cùng với môi trường benchmark; không nên coi một giá trị đo trên một máy cụ thể là đặc tính tuyệt đối của thuật toán.

---

## 18. Evaluation Protocol của Phase 5

Để tránh test-set contamination:

**Training benchmark:**
```text
retrieval_train.json
1,752 queries
```
được sử dụng cho:
- BM25 hyperparameter tuning.
- So sánh các cấu hình trong grid search.

**Hold-out test benchmark:**
```text
retrieval_test.json
438 queries
```
được giữ độc lập để đánh giá cuối cùng.

Tuning không được sử dụng test metrics để lựa chọn hyperparameters.

**Quy trình:**

```text
Retrieval Train
      |
      v
 Grid Search
      |
      v
Best BM25 configuration
      |
      v
Freeze configuration
      |
      v
 Hold-out Test
      |
      v
Final comparison
```

Điều này giúp kết quả Test phản ánh khả năng tổng quát hóa trên benchmark chưa được dùng để lựa chọn tham số.

---

## 19. Ý nghĩa của Baseline trong Project 2

TF-IDF và BM25 không phải mục tiêu cuối cùng của hệ thống. Chúng đóng vai trò:

- Thiết lập lexical baseline.
- Đo chất lượng retrieval trước dense retrieval.
- Kiểm tra mức độ hữu ích của lexical overlap trong Vietnamese legal text.
- Xác định các failure cases của sparse retrieval.
- Làm baseline để đánh giá các phương pháp dense retrieval ở Phase 6 trở đi.

Do đó, một dense retriever chỉ có ý nghĩa khi được so sánh công bằng với baseline đã được benchmark và freeze.

---

## 20. Liên hệ với các Phase tiếp theo

Kết quả của Phase 5 đặt ra hai câu hỏi quan trọng:

- **Câu hỏi 1 — Dense retrieval có vượt lexical retrieval không?**  
  Cần thực nghiệm bằng benchmark giống nhau và metric giống nhau.

- **Câu hỏi 2 — Dense retrieval có giải quyết được vocabulary mismatch không?**  
  Cần kiểm tra trực tiếp trên các failure cases của lexical baseline thay vì chỉ nhìn aggregate metrics.

Do đó Phase 5 không chỉ tạo ra một baseline về performance, mà còn cung cấp experimental reference point cho các phase retrieval tiếp theo.

---

## 21. Tóm tắt

| Thành phần | Vai trò |
| :--- | :--- |
| **Inverted Index** | Tăng hiệu quả truy cập postings |
| **TF-IDF** | Lexical ranking baseline |
| **BM25** | Lexical ranking với TF saturation + length normalization |
| **$k_1$** | Điều khiển TF saturation |
| **$b$** | Điều khiển document-length normalization |
| **Precision@k** | Độ chính xác Top-k |
| **Recall@k** | Coverage của relevant documents |
| **MRR@10** | Vị trí relevant result đầu tiên |
| **MAP@10/100** | Chất lượng ranking của nhiều relevant results |
| **nDCG@10/100** | Chất lượng ranking có discount theo vị trí |
| **Latency/QPS** | Hiệu năng hệ thống |
| **Train benchmark** | Tuning / model selection |
| **Test benchmark** | Hold-out final evaluation |

**Kết luận:** Phase 5 sử dụng TF-IDF và BM25 để thiết lập một lexical retrieval baseline có thể giải thích, tái lập và benchmark được. BM25 Tuned được lựa chọn dựa trên kết quả grid search trên Train, sau đó được đánh giá trên Hold-out Test. Các giới hạn của lexical matching sẽ được dùng làm cơ sở thực nghiệm để đánh giá Dense Retrieval ở các phase tiếp theo.