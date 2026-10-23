"""Bộ truy xuất lai (HybridRetriever) kết hợp dense vector và BM25 bằng RRF hoặc linear fusion."""

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Union
import pandas as pd

from src.retrieval.base import BaseRetriever, RetrievalResult


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        dense_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        fusion_method: str = "weighted_rrf",
        k_rrf: int = 10,
        w_dense: float = 0.80,
        w_bm25: float = 0.20,
        candidate_depth: int = 100,
    ) -> None:
        super().__init__()
        if fusion_method not in ("rrf", "weighted_rrf", "linear"):
            raise ValueError(
                f"Phương thức fusion '{fusion_method}' không hợp lệ. Chọn 'rrf', 'weighted_rrf', hoặc 'linear'."
            )
        if k_rrf <= 0:
            raise ValueError(f"Hằng số k_rrf phải > 0, nhận được: {k_rrf}")
        if w_dense < 0 or w_bm25 < 0:
            raise ValueError("Trọng số w_dense và w_bm25 phải không âm.")

        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.fusion_method = fusion_method
        self.k_rrf = int(k_rrf)
        self.w_dense = float(w_dense)
        self.w_bm25 = float(w_bm25)
        self.candidate_depth = int(candidate_depth)

        if getattr(dense_retriever, "is_fitted", False) and getattr(
            bm25_retriever, "is_fitted", False
        ):
            self.is_fitted = True

    def fit(self, corpus: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Self:
        if not getattr(self.dense_retriever, "is_fitted", False):
            self.dense_retriever.fit(corpus)
        if not getattr(self.bm25_retriever, "is_fitted", False):
            self.bm25_retriever.fit(corpus)
        self.is_fitted = True
        return self

    @staticmethod
    def _fuse_rrf(
        dense_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        k_rrf: int,
        w_dense: float,
        w_bm25: float,
        top_k: int,
    ) -> List[RetrievalResult]:
        fused_scores: Dict[int, float] = defaultdict(float)

        for res in dense_results:
            fused_scores[res.article_id] += w_dense / (k_rrf + res.rank)

        for res in bm25_results:
            fused_scores[res.article_id] += w_bm25 / (k_rrf + res.rank)

        sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            RetrievalResult(article_id=art_id, score=float(sc), rank=idx)
            for idx, (art_id, sc) in enumerate(sorted_items, start=1)
        ]

    @staticmethod
    def _fuse_linear(
        dense_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        alpha: float,
        top_k: int,
    ) -> List[RetrievalResult]:
        def min_max_norm(results: List[RetrievalResult]) -> Dict[int, float]:
            if not results:
                return {}
            scores = [r.score for r in results]
            min_s, max_s = min(scores), max(scores)
            denom = max_s - min_s
            if denom <= 1e-8:
                return {r.article_id: 1.0 for r in results}
            return {r.article_id: (r.score - min_s) / denom for r in results}

        norm_dense = min_max_norm(dense_results)
        norm_bm25 = min_max_norm(bm25_results)

        all_ids = set(norm_dense.keys()).union(norm_bm25.keys())
        fused_scores: Dict[int, float] = defaultdict(float)
        for aid in all_ids:
            s_dense = norm_dense.get(aid, 0.0)
            s_bm25 = norm_bm25.get(aid, 0.0)
            fused_scores[aid] = alpha * s_dense + (1.0 - alpha) * s_bm25

        sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            RetrievalResult(article_id=art_id, score=float(sc), rank=idx)
            for idx, (art_id, sc) in enumerate(sorted_items, start=1)
        ]

    def _fuse_single_query(
        self,
        dense_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        if self.fusion_method == "rrf":
            return self._fuse_rrf(
                dense_results, bm25_results, self.k_rrf, 0.5, 0.5, top_k
            )
        elif self.fusion_method == "weighted_rrf":
            return self._fuse_rrf(
                dense_results, bm25_results, self.k_rrf, self.w_dense, self.w_bm25, top_k
            )
        elif self.fusion_method == "linear":
            return self._fuse_linear(
                dense_results, bm25_results, self.w_dense, top_k
            )
        else:
            raise ValueError(f"Không hỗ trợ phương thức: {self.fusion_method}")

    def search(self, query: str, top_k: int = 100) -> List[RetrievalResult]:
        self._check_is_fitted()
        depth = max(top_k, self.candidate_depth)
        dense_res = self.dense_retriever.search(query, top_k=depth)
        bm25_res = self.bm25_retriever.search(query, top_k=depth)
        return self._fuse_single_query(dense_res, bm25_res, top_k)

    def batch_search(
        self, queries: List[str], top_k: int = 100
    ) -> List[List[RetrievalResult]]:
        self._check_is_fitted()
        depth = max(top_k, self.candidate_depth)
        dense_batch = self.dense_retriever.batch_search(queries, top_k=depth)
        bm25_batch = self.bm25_retriever.batch_search(queries, top_k=depth)

        fused_batch: List[List[RetrievalResult]] = []
        for dense_res, bm25_res in zip(dense_batch, bm25_batch):
            fused_batch.append(self._fuse_single_query(dense_res, bm25_res, top_k))
        return fused_batch

    def save(self, path: Union[str, Path]) -> None:
        self._check_is_fitted()
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "fusion_method": self.fusion_method,
            "k_rrf": self.k_rrf,
            "w_dense": self.w_dense,
            "w_bm25": self.w_bm25,
            "candidate_depth": self.candidate_depth,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        dense_retriever: Optional[BaseRetriever] = None,
        bm25_retriever: Optional[BaseRetriever] = None,
        **kwargs: Any,
    ) -> Self:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)

        return cls(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            fusion_method=config.get("fusion_method", "weighted_rrf"),
            k_rrf=config.get("k_rrf", 10),
            w_dense=config.get("w_dense", 0.80),
            w_bm25=config.get("w_bm25", 0.20),
            candidate_depth=config.get("candidate_depth", 100),
        )
