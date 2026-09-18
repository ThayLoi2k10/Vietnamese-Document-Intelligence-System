# src/retrieval/dense/retriever.py
"""Lớp triển khai DenseRetriever kế thừa BaseRetriever chuẩn của dự án.

Tuân thủ nghiêm ngặt đặc tả:
    - Kế thừa BaseRetriever
    - Trả về danh sách đối tượng RetrievalResult (article_id: int, score: float, rank: 1-indexed int)
    - Sắp xếp giảm dần theo điểm số mức độ liên quan (score)
    - Không có trùng lặp article_id
    - Hỗ trợ đầy đủ các phương thức fit, search, batch_search, save, load
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Union
import numpy as np
import pandas as pd

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.dense.encoder import (
    DEFAULT_ARTICLE_MAX_LENGTH,
    DEFAULT_QUERY_MAX_LENGTH,
    DenseEncoder,
)
from src.retrieval.dense.index import DenseIndex


class DenseRetriever(BaseRetriever):
    """Mô hình truy xuất văn bản ngữ nghĩa sử dụng Dense Bi-Encoder (BGE-M3)."""

    def __init__(
        self,
        encoder: Optional[DenseEncoder] = None,
        index: Optional[DenseIndex] = None,
        device: Optional[str] = None,
        use_fp16: Optional[bool] = None,
        query_max_length: int = DEFAULT_QUERY_MAX_LENGTH,
        article_max_length: int = DEFAULT_ARTICLE_MAX_LENGTH,
    ) -> None:
        super().__init__()
        self.device = device
        self.use_fp16 = use_fp16
        self.query_max_length = query_max_length
        self.article_max_length = article_max_length

        self.encoder = encoder
        self.index = index
        if self.index is not None and self.index.ntotal > 0:
            self.is_fitted = True

    def _get_or_init_encoder(self) -> DenseEncoder:
        """Khởi tạo encoder nếu chưa được nạp sẵn."""
        if self.encoder is None:
            self.encoder = DenseEncoder(
                device=self.device,
                use_fp16=self.use_fp16,
                query_max_length=self.query_max_length,
                article_max_length=self.article_max_length,
            )
        return self.encoder

    def fit(self, corpus: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Self:
        """Nạp corpus, mã hóa toàn bộ văn bản và xây dựng FAISS Flat IP Index.

        Args:
            corpus: DataFrame hoặc danh sách bản ghi chứa các trường:
                    - 'global_article_id' hoặc 'article_id'
                    - 'text' hoặc 'article_content' hoặc 'content'

        Returns:
            self: Đối tượng DenseRetriever sau khi đã xây dựng xong chỉ mục.
        """
        if isinstance(corpus, list):
            df = pd.DataFrame(corpus)
        else:
            df = corpus

        # Xác định cột ID
        if "global_article_id" in df.columns:
            id_col = "global_article_id"
        elif "article_id" in df.columns:
            id_col = "article_id"
        else:
            raise KeyError("Không tìm thấy cột 'global_article_id' hoặc 'article_id'.")

        # Xác định cột văn bản
        if "text" in df.columns:
            text_col = "text"
        elif "article_content" in df.columns:
            text_col = "article_content"
        elif "content" in df.columns:
            text_col = "content"
        else:
            raise KeyError("Không tìm thấy cột nội dung văn bản.")

        article_ids = df[id_col].astype(int).tolist()
        texts = df[text_col].fillna("").astype(str).tolist()

        encoder = self._get_or_init_encoder()
        embeddings = encoder.encode_articles(texts)

        self.index = DenseIndex(dimension=embeddings.shape[1])
        self.index.add(embeddings, article_ids)
        self.is_fitted = True

        return self

    def search(self, query_text: str, top_k: int = 10) -> List[RetrievalResult]:
        """Tìm kiếm Top-k điều luật liên quan cho một câu truy vấn.

        Args:
            query_text: Chuỗi câu hỏi / truy vấn.
            top_k: Số lượng kết quả cần lấy (top_k > 0).

        Returns:
            Danh sách RetrievalResult (1-indexed rank, sắp xếp giảm dần theo score).
        """
        self._check_is_fitted()
        assert self.index is not None

        encoder = self._get_or_init_encoder()
        query_vec = encoder.encode_queries([query_text], batch_size=1)

        scores, doc_ids = self.index.search(query_vec, top_k=top_k)

        results: List[RetrievalResult] = []
        if len(doc_ids) > 0:
            for rank_idx, (art_id, sc) in enumerate(zip(doc_ids[0], scores[0]), start=1):
                results.append(
                    RetrievalResult(
                        article_id=int(art_id),
                        score=float(sc),
                        rank=rank_idx,
                    )
                )

        return results

    def batch_search(
        self, queries: List[str], top_k: int = 10, batch_size: int = 32
    ) -> List[List[RetrievalResult]]:
        """Truy vấn hàng loạt nhằm tối ưu hóa tốc độ đánh giá benchmark.

        Args:
            queries: Danh sách các câu truy vấn.
            top_k: Số lượng kết quả tối đa cho mỗi query.
            batch_size: Kích thước batch khi mã hóa query.

        Returns:
            Danh sách kết quả tương ứng với từng query đầu vào.
        """
        self._check_is_fitted()
        assert self.index is not None

        if len(queries) == 0:
            return []

        encoder = self._get_or_init_encoder()
        query_vecs = encoder.encode_queries(queries, batch_size=batch_size)

        scores_mat, doc_ids_list = self.index.search(query_vecs, top_k=top_k)

        all_results: List[List[RetrievalResult]] = []
        for q_idx in range(len(queries)):
            q_results: List[RetrievalResult] = []
            for rank_idx, (art_id, sc) in enumerate(
                zip(doc_ids_list[q_idx], scores_mat[q_idx]), start=1
            ):
                q_results.append(
                    RetrievalResult(
                        article_id=int(art_id),
                        score=float(sc),
                        rank=rank_idx,
                    )
                )
            all_results.append(q_results)

        return all_results

    def save(self, path: Union[str, Path]) -> None:
        """Lưu chỉ mục FAISS và danh sách ID xuống thư mục chỉ định."""
        self._check_is_fitted()
        assert self.index is not None

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        index_file = save_dir / "index.faiss"
        ids_file = save_dir / "corpus_ids.json"
        self.index.save(index_file, ids_file)

    @classmethod
    def load(cls, path: Union[str, Path], encoder: Optional[DenseEncoder] = None) -> Self:
        """Nạp chỉ mục và khôi phục retriever từ thư mục đã lưu."""
        load_dir = Path(path)
        index_file = load_dir / "index.faiss"
        ids_file = load_dir / "corpus_ids.json"

        dense_index = DenseIndex.load(index_file, ids_file)
        retriever = cls(encoder=encoder, index=dense_index)
        retriever.is_fitted = True
        return retriever
