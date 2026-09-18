# src/rag/reranking/adapters/cross_encoder_reranker.py
"""Adapter cho các mô hình Cross-Encoder Reranker (BGE-Reranker-v2-M3, ViRanker).

Tuân thủ Mục 9:
    - 9.3 Candidate models [HYP]: Không hard-code winner logic
    - 9.4 Reranker contract: Bảo toàn provenance, tính điểm và rank mới
"""

from typing import Any, Callable, Dict, List, Optional

from src.rag.errors.exceptions import IncompatibleComponentVersionError
from src.rag.reranking.domain.models import (
    RankedCandidate,
    RankedResults,
    RerankOptions,
)
from src.rag.reranking.interfaces.reranker import Reranker
from src.rag.retrieval.domain.models import RetrievalItem


class CrossEncoderRerankerAdapter(Reranker):
    """Adapter tích hợp các mô hình Cross-Encoder vào Phase 9 Reranking interface."""

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-reranker-v2-m3",
        predict_fn: Optional[Callable[[List[List[str]]], List[float]]] = None,
        component_version: str = "v1.0.0",
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.predict_fn = predict_fn
        self.component_version = component_version
        self.device = device
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalItem],
        options: RerankOptions,
    ) -> RankedResults:
        top_k = options.reranker_top_k
        selected_candidates = candidates[:top_k]

        if not selected_candidates:
            return RankedResults(
                query=query,
                candidates=[],
                reranker_metadata={
                    "model_name": self.model_name_or_path,
                    "component_version": self.component_version,
                    "top_k": top_k,
                },
            )

        pairs = [[query, item.text] for item in selected_candidates]

        if self.predict_fn is not None:
            scores = self.predict_fn(pairs)
        else:
            # Fallback nếu không có predict_fn (ví dụ trong môi trường offline/unfitted)
            scores = [
                float(item.score if item.score is not None else 0.0)
                for item in selected_candidates
            ]

        # Ghép cặp và sắp xếp giảm dần theo điểm rerank
        scored_pairs = list(zip(selected_candidates, scores))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        ranked_candidates: List[RankedCandidate] = []
        for new_rank, (item, sc) in enumerate(scored_pairs, start=1):
            ranked_candidates.append(
                RankedCandidate(
                    retrieval_item=item,
                    retrieval_rank=item.rank,
                    rerank_rank=new_rank,
                    retrieval_score=item.score,
                    rerank_score=float(sc),
                )
            )

        return RankedResults(
            query=query,
            candidates=ranked_candidates,
            reranker_metadata={
                "model_name": self.model_name_or_path,
                "component_version": self.component_version,
                "top_k": top_k,
                "input_count": len(candidates),
                "output_count": len(ranked_candidates),
            },
        )
