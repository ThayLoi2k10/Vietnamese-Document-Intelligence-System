"""Bộ truy xuất cấp độ chunk và tổng hợp max-score về cấp độ điều luật."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Union
import pandas as pd

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.dense.encoder import DenseEncoder
from src.retrieval.indexing.base import BaseVectorStore
from src.retrieval.indexing.faiss_vector_store import FaissVectorStore
from src.retrieval.indexing.qdrant_store import QdrantVectorStore


class ChunkedVectorRetriever(BaseRetriever):
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

    @staticmethod
    def _aggregate_article_max_scores(
        chunk_results: List[Any], top_k: int
    ) -> List[RetrievalResult]:
        article_max_score: Dict[int, float] = {}
        for r in chunk_results:
            art_id = r.global_article_id
            if art_id >= 0 and r.score > article_max_score.get(art_id, -1e9):
                article_max_score[art_id] = r.score

        sorted_articles = sorted(
            article_max_score.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        return [
            RetrievalResult(article_id=int(art_id), score=float(sc), rank=rank_idx)
            for rank_idx, (art_id, sc) in enumerate(sorted_articles, start=1)
        ]

    def fit(self, corpus: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Self:
        self.is_fitted = True
        return self

    def search(self, query_text: str, top_k: int = 10) -> List[RetrievalResult]:
        self._check_is_fitted()
        encoder = self._get_or_init_encoder()
        q_vec = encoder.encode_queries([query_text], batch_size=1)
        chunk_results = self.vector_store.search(q_vec, limit=self.chunk_depth)
        return self._aggregate_article_max_scores(chunk_results, top_k)

    def batch_search(
        self, queries: List[str], top_k: int = 10, batch_size: int = 32
    ) -> List[List[RetrievalResult]]:
        self._check_is_fitted()
        if not queries:
            return []

        encoder = self._get_or_init_encoder()
        query_vecs = encoder.encode_queries(queries, batch_size=batch_size)

        all_results: List[List[RetrievalResult]] = []
        for i in range(len(queries)):
            q_vec = query_vecs[i : i + 1]
            chunk_results = self.vector_store.search(q_vec, limit=self.chunk_depth)
            all_results.append(self._aggregate_article_max_scores(chunk_results, top_k))

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
