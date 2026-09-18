# src/retrieval/chunked_retriever.py
"""Bộ truy xuất văn bản ChunkedVectorRetriever kế thừa BaseRetriever chuẩn của dự án.

Tuân thủ nghiêm ngặt:
    - Kế thừa BaseRetriever từ src/retrieval/base.py
    - Trả về danh sách đối tượng RetrievalResult (article_id: int, score: float, rank: 1-indexed int)
    - Tích hợp vector store (Qdrant hoặc FAISS) với cơ chế tổng hợp max-score điều luật
    - Tương thích 100% với hệ thống đánh giá IR metrics hiện có của dự án
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Union
import numpy as np
import pandas as pd

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.dense.encoder import DenseEncoder
from src.retrieval.indexing.base import BaseVectorStore
from src.retrieval.indexing.faiss_vector_store import FaissVectorStore
from src.retrieval.indexing.qdrant_store import QdrantVectorStore


class ChunkedVectorRetriever(BaseRetriever):
    """Retriever thực thi tìm kiếm ở cấp độ Chunk và tổng hợp Max-score về cấp độ Article."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        encoder: Optional[DenseEncoder] = None,
        chunk_depth: int = 300,
    ) -> None:
        super().__init__()
        self.vector_store = vector_store
        self.encoder = encoder
        self.chunk_depth = chunk_depth
        if self.vector_store.count() > 0:
            self.is_fitted = True

    def _get_or_init_encoder(self) -> DenseEncoder:
        if self.encoder is None:
            self.encoder = DenseEncoder()
        return self.encoder

    def fit(self, corpus: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Self:
        """Đã được nạp sẵn qua quy trình Phase 7/8 indexing."""
        self.is_fitted = True
        return self

    def search(self, query_text: str, top_k: int = 10) -> List[RetrievalResult]:
        """Truy vấn Top-k điều luật liên quan thông qua tìm kiếm chunk và gom nhóm max-score."""
        self._check_is_fitted()
        encoder = self._get_or_init_encoder()
        q_vec = encoder.encode_queries([query_text], batch_size=1)

        # 1. Tìm kiếm chunk
        chunk_results = self.vector_store.search(q_vec, limit=self.chunk_depth)

        # 2. Tổng hợp max-score theo global_article_id
        article_max_score: Dict[int, float] = {}
        for r in chunk_results:
            art_id = r.global_article_id
            if art_id >= 0:
                if r.score > article_max_score.get(art_id, -1e9):
                    article_max_score[art_id] = r.score

        # 3. Xếp hạng giảm dần
        sorted_articles = sorted(
            article_max_score.items(), key=lambda x: x[1], reverse=True
        )

        results: List[RetrievalResult] = []
        for rank_idx, (art_id, sc) in enumerate(sorted_articles[:top_k], start=1):
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
        """Truy vấn hàng loạt phục vụ đánh giá benchmark thông lượng cao."""
        self._check_is_fitted()
        if not queries:
            return []

        encoder = self._get_or_init_encoder()
        query_vecs = encoder.encode_queries(queries, batch_size=batch_size)

        all_results: List[List[RetrievalResult]] = []
        for i in range(len(queries)):
            q_vec = query_vecs[i : i + 1]
            chunk_results = self.vector_store.search(q_vec, limit=self.chunk_depth)

            article_max_score: Dict[int, float] = {}
            for r in chunk_results:
                art_id = r.global_article_id
                if art_id >= 0:
                    if r.score > article_max_score.get(art_id, -1e9):
                        article_max_score[art_id] = r.score

            sorted_articles = sorted(
                article_max_score.items(), key=lambda x: x[1], reverse=True
            )

            q_results: List[RetrievalResult] = []
            for rank_idx, (art_id, sc) in enumerate(sorted_articles[:top_k], start=1):
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
        self._check_is_fitted()
        self.vector_store.save(path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        encoder: Optional[DenseEncoder] = None,
        store_type: str = "qdrant",
    ) -> Self:
        load_path = Path(path)
        if store_type.lower() == "qdrant":
            store = QdrantVectorStore.load(load_path)
        else:
            store = FaissVectorStore.load(load_path)
        retriever = cls(vector_store=store, encoder=encoder)
        retriever.is_fitted = True
        return retriever
