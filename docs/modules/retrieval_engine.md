# Module: Retrieval Engine (`src/retrieval/`)

## 1. Tổng quan & Trách nhiệm (What does it do?)
Gói module [`src/retrieval/`](file:///d:/PROJECT/PROJECT%202/src/retrieval/) chịu trách nhiệm toàn bộ các thuật toán và cấu trúc dữ liệu liên quan đến việc lập chỉ mục, lưu trữ và truy xuất thông tin pháp lý từ 59.636 điều luật (`global_article_id` từ `0` đến `59635`).

* **Nguyên tắc cốt lõi:**
  * Kế thừa từ giao thức trừu tượng thống nhất `BaseRetriever`.
  * Trả về kết quả chuẩn hóa theo `list[RetrievalResult]` (1-indexed rank, article_id, score).
  * Không làm thay đổi dải không gian ID cố định của hệ thống.

---

## 2. Bản đồ Cấu trúc Module (Package Map)

```text
src/retrieval/
├── base.py                   # Abstract Base Class BaseRetriever & RetrievalResult dataclass
├── bm25_retriever.py         # Okapi BM25 Inverted Index với NumPy scoring vector hóa
├── tfidf_retriever.py        # Scikit-learn Sparse TF-IDF Vectorizer
├── dense_hf_retriever.py     # HFDenseRetriever (BGE-M3 dùng thuần transformers + FAISS, tránh phụ thuộc FlagEmbedding)
├── chunked_retriever.py      # Retriever bọc dữ liệu phân đoạn (Chunks) kèm Max-pooling
├── dense/                    # Tầng truy xuất Dense Bi-Encoder
│   ├── encoder.py            # BGE-M3 Dense Embedding extractor (1024-d, L2 normalized)
│   ├── index.py              # FAISS IndexFlatIP Wrapper (Inner Product = Cosine Similarity)
│   └── retriever.py          # DenseRetriever kế thừa BaseRetriever
├── hybrid/                   # Tầng truy xuất hỗn hợp
│   ├── fusion.py             # Giải thuật Weighted Reciprocal Rank Fusion (wRRF)
│   └── retriever.py          # HybridRetriever kết hợp BM25 + BGE-M3
├── chunking/                 # Bộ phân đoạn cấu trúc điều luật
│   ├── parser.py             # Regex Parser phân cấp: Điều -> Khoản -> Điểm
│   ├── hierarchical_chunker.py # Chunker phân tầng 512 tokens (Chiến lược C9)
│   └── chunk_record.py       # Dataclass đại diện cho từng chunk
├── indexing/                 # Quản trị Vector Database Qdrant
│   ├── qdrant_store.py       # Client Qdrant Local Persistent Collection
│   ├── point_id.py           # Sinh UUID v5 tất định từ chunk_id
│   ├── reconciliation.py     # Đối soát tính toàn vẹn (Missing, Stale, Orphan vectors)
│   └── rebuild.py            # Khôi phục & Rebuild chỉ mục
└── reranking/                # Tầng Cross-Encoder Reranker
    ├── models.py             # Data models cho reranker candidates
    ├── reranker.py           # FlagReranker wrapper cho BAAI/bge-reranker-base (Zero-shot)
    └── finetuned_reranker.py # FinetunedLegalReranker (LoRA r=16 + Residual Hybrid Fusion α=0.45)
```

---

## 3. Các Class & Phương thức Then chốt (Key APIs)

### A. `BaseRetriever` & `RetrievalResult` (`src/retrieval/base.py`)
```python
@dataclass
class RetrievalResult:
    article_id: int   # Định danh điều luật toàn cục [0..59635]
    score: float      # Điểm số truy xuất (BM25, Cosine, hoặc wRRF score)
    rank: int         # Thứ hạng trả về (1-indexed)

class BaseRetriever(ABC):
    @abstractmethod
    def fit(self, corpus: pd.DataFrame) -> "BaseRetriever": ...
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]: ...
    
    @abstractmethod
    def save(self, path: Path | str) -> None: ...
    
    @classmethod
    @abstractmethod
    def load(cls, path: Path | str) -> "BaseRetriever": ...
```

### B. `HybridRetriever` (`src/retrieval/hybrid/retriever.py`)
* **Trách nhiệm:** Production Retriever chính thức của hệ thống.
* **Cơ chế:** Kích hoạt song song `dense_retriever.search(query, top_k=100)` và `bm25_retriever.search(query, top_k=100)`, sau đó gọi `weighted_reciprocal_rank_fusion` để tính điểm tổng hợp:
  $$\text{Score}(d) = 0.75 \cdot \frac{1}{5 + r_{\text{dense}}(d)} + 0.25 \cdot \frac{1}{5 + r_{\text{bm25}}(d)}$$

### C. `HFDenseRetriever` (`src/retrieval/dense_hf_retriever.py`) & `FinetunedLegalReranker` (`src/retrieval/reranking/finetuned_reranker.py`)
* **`HFDenseRetriever`:** Lớp truy xuất ngữ nghĩa BGE-M3 sử dụng thuần `transformers` (`AutoTokenizer`, `AutoModel`) và `faiss.IndexFlatIP`, hỗ trợ nạp trực tiếp cả `embeddings.f32.npy` lẫn `index.faiss` mà không cần `FlagEmbedding` / `datasets`.
* **`FinetunedLegalReranker`:** Bộ tái xếp hạng Cross-Encoder đã huấn luyện chuyên biệt cho Pháp luật Việt Nam, gộp trực tiếp 72 ma trận LoRA (`r=16, alpha=32`) và `classifier` head từ `data/indices/reranker_finetuned/lora_reranker_weights.pt` vào `BAAI/bge-reranker-base`, đồng thời kết hợp điểm ưu tiên Tầng 1 theo công thức Residual Hybrid Fusion ($\alpha = 0.45$), đạt **Hit@1 = 57.76%, Hit@5 = 84.47%, nDCG@10 = 0.6911** trên tập Frozen Test (438 truy vấn).

### D. `QdrantVectorStore` (`src/retrieval/indexing/qdrant_store.py`)
* **Trách nhiệm:** Lưu trữ persistent vector 1024 chiều cục bộ tại `data/indices/qdrant_legal/`.
* **Phương thức chính:**
  * `upsert_batch(points: list[VectorPoint])`: Nạp lặp an toàn (idempotent) với point_id dạng UUID v5.
  * `search(query_vector, top_k, filters)`: Tìm kiếm cosine kết hợp lọc trước (pre-filter) theo `law_id` hoặc cờ `is_active`.

---

## 4. Hướng dẫn Lập trình viên Sử dụng (Developer Usage)

```python
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense.retriever import DenseRetriever
from src.retrieval.hybrid.retriever import HybridRetriever

# 1. Khởi tạo và nạp các chỉ mục đã huấn luyện
bm25 = BM25Retriever.load("data/indices/bm25_index.pkl")
bm25.set_params(k1=1.80, b=0.85)

dense = DenseRetriever.load(
    index_path="data/indices/dense_index_bge_m3.faiss",
    model_name_or_path="BAAI/bge-m3"
)

# 2. Đóng gói thành Hybrid Retriever cho Production
hybrid = HybridRetriever(
    dense_retriever=dense,
    bm25_retriever=bm25,
    k_rrf=5,
    w_dense=0.75,
    w_bm25=0.25
)

# 3. Tìm kiếm văn bản
results = hybrid.search("Điều kiện hưởng chế độ thai sản đối với lao động nữ?", top_k=5)
for r in results:
    print(f"Hạng {r.rank}: Điều luật {r.article_id} - Điểm wRRF: {r.score:.5f}")
```
