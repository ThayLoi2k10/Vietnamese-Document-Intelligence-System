"""Mô hình truy xuất thông tin TF-IDF dựa trên cosine similarity và ma trận thưa."""

from pathlib import Path
from typing import Any, Dict, List, Self, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.retrieval.base import BaseRetriever, RetrievalResult, extract_corpus_columns


class TfidfRetriever(BaseRetriever):
    def __init__(
        self,
        vectorizer: TfidfVectorizer | None = None,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int = 1,
        max_df: float = 0.9,
    ) -> None:
        super().__init__()
        self.vectorizer = vectorizer or TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            norm="l2",
            sublinear_tf=True,
        )
        self.article_ids: List[int] = []
        self.tfidf_matrix = None

    def fit(self, corpus: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Self:
        self.article_ids, texts = extract_corpus_columns(corpus)
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return self

    def search(self, query_text: str, top_k: int = 10) -> List[RetrievalResult]:
        self._check_is_fitted()

        if top_k <= 0:
            raise ValueError("top_k phải lớn hơn 0.")

        if not query_text or not query_text.strip():
            return []

        query_vec = self.vectorizer.transform([query_text])
        scores = (self.tfidf_matrix.dot(query_vec.T)).toarray().ravel()

        if len(scores) == 0:
            return []

        actual_k = min(top_k, len(scores))
        top_k_indices = np.argpartition(scores, -actual_k)[-actual_k:]
        sorted_indices = top_k_indices[np.argsort(-scores[top_k_indices])]

        return [
            RetrievalResult(
                article_id=self.article_ids[idx],
                score=float(scores[idx]),
                rank=rank,
            )
            for rank, idx in enumerate(sorted_indices, start=1)
        ]

    def batch_search(
        self, queries: List[str], top_k: int = 10, batch_size: int = 512
    ) -> List[List[RetrievalResult]]:
        self._check_is_fitted()

        if top_k <= 0:
            raise ValueError("top_k phải lớn hơn 0.")
        if batch_size <= 0:
            raise ValueError("batch_size phải lớn hơn 0.")

        if not queries:
            return []

        num_docs = len(self.article_ids)
        if num_docs == 0:
            return [[] for _ in queries]
        actual_k = min(top_k, num_docs)
        all_results: List[List[RetrievalResult]] = []

        for start_idx in range(0, len(queries), batch_size):
            batch_queries = queries[start_idx : start_idx + batch_size]
            query_matrix = self.vectorizer.transform(batch_queries)
            scores_chunk = (query_matrix.dot(self.tfidf_matrix.T)).toarray()

            for q_sub_idx, q_text in enumerate(batch_queries):
                if not q_text or not q_text.strip():
                    all_results.append([])
                    continue

                scores = scores_chunk[q_sub_idx]
                top_k_indices = np.argpartition(scores, -actual_k)[-actual_k:]
                sorted_indices = top_k_indices[np.argsort(-scores[top_k_indices])]

                q_results = [
                    RetrievalResult(
                        article_id=self.article_ids[idx],
                        score=float(scores[idx]),
                        rank=rank,
                    )
                    for rank, idx in enumerate(sorted_indices, start=1)
                ]
                all_results.append(q_results)

        return all_results

    def save(self, path: Union[str, Path]) -> None:
        self._check_is_fitted()
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vectorizer": self.vectorizer,
            "article_ids": self.article_ids,
            "tfidf_matrix": self.tfidf_matrix,
        }
        temp_path = save_path.with_suffix(save_path.suffix + ".tmp")
        try:
            joblib.dump(payload, temp_path, compress=3)
            temp_path.replace(save_path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Union[str, Path]) -> Self:
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Không tìm thấy index tại: {load_path}")
        payload = joblib.load(load_path)
        instance = cls()
        instance.vectorizer = payload["vectorizer"]
        instance.article_ids = payload["article_ids"]
        instance.tfidf_matrix = payload["tfidf_matrix"]
        instance.is_fitted = True
        return instance