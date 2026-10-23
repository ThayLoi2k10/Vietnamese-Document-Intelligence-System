"""Mô hình truy xuất văn bản ngữ nghĩa sử dụng dense bi-encoder và FAISS flat index."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Union
import pandas as pd

from src.retrieval.base import BaseRetriever, RetrievalResult, extract_corpus_columns
from src.retrieval.dense.encoder import (
    DEFAULT_ARTICLE_MAX_LENGTH,
    DEFAULT_QUERY_MAX_LENGTH,
    DenseEncoder,
)
from src.retrieval.dense.index import DenseIndex


class DenseRetriever(BaseRetriever):
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
        if self.encoder is None:
            self.encoder = DenseEncoder(
                device=self.device,
                use_fp16=self.use_fp16,
                query_max_length=self.query_max_length,
                article_max_length=self.article_max_length,
            )
        return self.encoder

    def fit(self, corpus: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Self:
        article_ids, texts = extract_corpus_columns(corpus)
        encoder = self._get_or_init_encoder()
        embeddings = encoder.encode_articles(texts)

        self.index = DenseIndex(dimension=embeddings.shape[1])
        self.index.add(embeddings, article_ids)
        self.is_fitted = True
        return self

    def search(self, query_text: str, top_k: int = 10) -> List[RetrievalResult]:
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
        self._check_is_fitted()
        assert self.index is not None

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.index.save(save_dir / "index.faiss", save_dir / "corpus_ids.json")

    @classmethod
    def load(cls, path: Union[str, Path], encoder: Optional[DenseEncoder] = None) -> Self:
        load_dir = Path(path)
        dense_index = DenseIndex.load(load_dir / "index.faiss", load_dir / "corpus_ids.json")
        retriever = cls(encoder=encoder, index=dense_index)
        retriever.is_fitted = True
        return retriever
