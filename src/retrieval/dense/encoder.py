# src/retrieval/dense/encoder.py
"""Bộ mã hóa Dense Embedding cho Phase 6 sử dụng BAAI/bge-m3.

Tuân thủ nghiêm ngặt đặc tả:
    - Mô hình: BAAI/bge-m3
    - Revision: 5617a9f61b028005a4858fdac845db406aefb181
    - Chế độ: dense-only (native CLS representation)
    - Kích thước vector: 1024 chiều
    - Chuẩn hóa: L2 normalization
    - Kiểu dữ liệu trả về: float32 numpy array
    - Bảo toàn vị trí: 100% tài liệu kể cả tài liệu rỗng (không xóa/bỏ dòng)
    - Kiểm soát lỗi: Reject zero vector (fail-fast)
"""

from typing import List, Optional, Union
import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel

OFFICIAL_MODEL_NAME = "BAAI/bge-m3"
OFFICIAL_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EMBEDDING_DIMENSION = 1024
DEFAULT_QUERY_MAX_LENGTH = 512
DEFAULT_ARTICLE_MAX_LENGTH = 2048


class DenseEncoder:
    """Encoder mã hóa câu truy vấn và văn bản điều luật sang vector 1024 chiều."""

    def __init__(
        self,
        model_name_or_path: str = OFFICIAL_MODEL_NAME,
        model_revision: str = OFFICIAL_MODEL_REVISION,
        device: Optional[str] = None,
        use_fp16: Optional[bool] = None,
        query_max_length: int = DEFAULT_QUERY_MAX_LENGTH,
        article_max_length: int = DEFAULT_ARTICLE_MAX_LENGTH,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.model_revision = model_revision
        self.query_max_length = query_max_length
        self.article_max_length = article_max_length
        self.normalize_embeddings = normalize_embeddings

        # Tự động chọn device nếu không chỉ định
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # FP16 chỉ kích hoạt trên GPU CUDA
        if use_fp16 is None:
            self.use_fp16 = True if "cuda" in self.device else False
        else:
            self.use_fp16 = use_fp16 and ("cuda" in self.device)

        # Khởi tạo mô hình BGE-M3
        self.model = BGEM3FlagModel(
            model_name_or_path=self.model_name_or_path,
            normalize_embeddings=self.normalize_embeddings,
            use_fp16=self.use_fp16,
            devices=self.device,
            pooling_method="cls",
            query_instruction_for_retrieval=None,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

    def _validate_embeddings(self, embeddings: np.ndarray, expected_count: int) -> np.ndarray:
        """Kiểm tra tính toàn vẹn của ma trận embedding đầu ra."""
        if embeddings.ndim != 2:
            raise ValueError(f"Ma trận embedding phải có 2 chiều, nhận được: {embeddings.shape}")
        if embeddings.shape[0] != expected_count:
            raise ValueError(
                f"Số lượng vector ({embeddings.shape[0]}) không khớp với đầu vào ({expected_count})."
            )
        if embeddings.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Kích thước vector ({embeddings.shape[1]}) không khớp với chuẩn {EMBEDDING_DIMENSION}."
            )

        # Chuyển đổi dứt khoát sang float32
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Kiểm tra zero vector (Fail-fast)
        norms = np.linalg.norm(embeddings, axis=1)
        zero_mask = (norms == 0.0) | np.isnan(norms)
        if np.any(zero_mask):
            bad_indices = np.where(zero_mask)[0].tolist()
            raise ValueError(
                f"Phát hiện vector rỗng hoặc NaN tại các vị trí: {bad_indices[:5]} "
                f"(tổng số: {len(bad_indices)})."
            )

        return embeddings

    def encode_queries(
        self,
        queries: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Mã hóa danh sách câu truy vấn thành ma trận dense vector [N, 1024].

        Args:
            queries: Chuỗi văn bản hoặc danh sách câu hỏi truy vấn.
            batch_size: Kích thước batch suy luận.
            show_progress_bar: Hiển thị thanh tiến trình nếu cần.

        Returns:
            np.ndarray kiểu float32, kích thước [N, 1024], đã chuẩn hóa L2.
        """
        if isinstance(queries, str):
            query_list = [queries]
        else:
            query_list = list(queries)

        if len(query_list) == 0:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

        # Thay thế None bằng chuỗi rỗng để tránh crash tokenizer
        safe_queries = [q if q is not None else "" for q in query_list]

        output = self.model.encode(
            safe_queries,
            batch_size=batch_size,
            max_length=self.query_max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense_vecs = output["dense_vecs"]
        return self._validate_embeddings(dense_vecs, len(safe_queries))

    def encode_articles(
        self,
        articles: Union[str, List[str]],
        batch_size: int = 4,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Mã hóa văn bản các điều luật thành ma trận dense vector [N, 1024].

        Args:
            articles: Chuỗi văn bản hoặc danh sách nội dung điều luật.
            batch_size: Kích thước batch suy luận.
            show_progress_bar: Hiển thị thanh tiến trình.

        Returns:
            np.ndarray kiểu float32, kích thước [N, 1024], đã chuẩn hóa L2.
        """
        if isinstance(articles, str):
            article_list = [articles]
        else:
            article_list = list(articles)

        if len(article_list) == 0:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

        # Xử lý các điều luật rỗng: bảo toàn vị trí hàng, không được loại bỏ
        safe_articles = [a if a is not None else "" for a in article_list]

        output = self.model.encode(
            safe_articles,
            batch_size=batch_size,
            max_length=self.article_max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense_vecs = output["dense_vecs"]
        return self._validate_embeddings(dense_vecs, len(safe_articles))
