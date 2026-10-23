"""Bộ mã hóa dense embedding cho câu truy vấn và điều luật sử dụng BAAI/bge-m3."""

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

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if use_fp16 is None:
            self.use_fp16 = "cuda" in self.device
        else:
            self.use_fp16 = use_fp16 and ("cuda" in self.device)

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

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

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
        query_list = [queries] if isinstance(queries, str) else list(queries)
        if len(query_list) == 0:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

        safe_queries = [q if q is not None else "" for q in query_list]
        output = self.model.encode(
            safe_queries,
            batch_size=batch_size,
            max_length=self.query_max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return self._validate_embeddings(output["dense_vecs"], len(safe_queries))

    def encode_articles(
        self,
        articles: Union[str, List[str]],
        batch_size: int = 4,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        article_list = [articles] if isinstance(articles, str) else list(articles)
        if len(article_list) == 0:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

        safe_articles = [a if a is not None else "" for a in article_list]
        output = self.model.encode(
            safe_articles,
            batch_size=batch_size,
            max_length=self.article_max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return self._validate_embeddings(output["dense_vecs"], len(safe_articles))
