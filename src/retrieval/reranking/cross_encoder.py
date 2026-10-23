"""Bộ tái xếp hạng hai giai đoạn cho văn bản pháp luật sử dụng mô hình cross-encoder."""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Sequence

from src.retrieval.base import RetrievalResult

logger = logging.getLogger(__name__)


class ArticleReranker:
    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None,
        use_fp16: Optional[bool] = None,
        batch_size: int = 32,
        max_length: int = 512,
        custom_predict_fn: Optional[Callable[[List[List[str]]], List[float]]] = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.max_length = max_length
        self.custom_predict_fn = custom_predict_fn
        self._model = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        if self.custom_predict_fn is not None:
            return None

        try:
            import torch
            from FlagEmbedding import FlagReranker

            fp16_enabled = torch.cuda.is_available() if self.use_fp16 is None else self.use_fp16
            self._model = FlagReranker(
                self.model_name_or_path,
                use_fp16=fp16_enabled,
                device=self.device,
                max_length=self.max_length,
            )
            return self._model
        except ImportError as e:
            raise ImportError(
                "FlagEmbedding is required for ArticleReranker. Install it via pip install FlagEmbedding."
            ) from e

    def compute_scores(self, pairs: Sequence[Sequence[str]]) -> List[float]:
        if not pairs:
            return []

        if self.custom_predict_fn is not None:
            return [float(s) for s in self.custom_predict_fn(list(pairs))]

        model = self._load_model()
        pair_list = [[str(q), str(d)] for q, d in pairs]
        raw_scores = model.compute_score(
            pair_list,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )

        if isinstance(raw_scores, (int, float)):
            return [float(raw_scores)]
        return [float(s) for s in raw_scores]

    def rerank(
        self,
        query: str,
        candidate_ids: Sequence[int],
        candidate_texts: Sequence[str],
        candidate_pool_size: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        if len(candidate_ids) != len(candidate_texts):
            raise ValueError(
                f"Length mismatch: {len(candidate_ids)} IDs vs {len(candidate_texts)} texts."
            )

        if not candidate_ids:
            return []

        pool_size = (
            min(candidate_pool_size, len(candidate_ids))
            if candidate_pool_size is not None
            else len(candidate_ids)
        )

        pool_ids = list(candidate_ids[:pool_size])
        pool_texts = list(candidate_texts[:pool_size])
        remaining_ids = list(candidate_ids[pool_size:])

        pairs = [[query, text] for text in pool_texts]
        scores = self.compute_scores(pairs)

        scored_pool = list(zip(pool_ids, scores))
        scored_pool.sort(key=lambda item: item[1], reverse=True)

        results: List[RetrievalResult] = []
        for rank, (art_id, score) in enumerate(scored_pool, start=1):
            results.append(RetrievalResult(article_id=art_id, score=score, rank=rank))

        min_pool_score = min(scores) if scores else 0.0
        start_rank = len(results) + 1
        for i, rem_id in enumerate(remaining_ids):
            rem_rank = start_rank + i
            rem_score = min_pool_score - 1.0 - (i * 0.01)
            results.append(
                RetrievalResult(article_id=rem_id, score=rem_score, rank=rem_rank)
            )

        if top_k is not None:
            results = results[:top_k]

        return results
