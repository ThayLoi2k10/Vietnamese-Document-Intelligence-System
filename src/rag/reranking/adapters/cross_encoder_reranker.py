"""Cross-encoder adapter for re-ranking retrieved candidates."""

from typing import Callable, List, Optional

from src.rag.reranking.domain.models import (
    RankedCandidate,
    RankedResults,
    RerankOptions,
)
from src.rag.reranking.interfaces.reranker import Reranker
from src.rag.retrieval.domain.models import RetrievalItem


class CrossEncoderRerankerAdapter(Reranker):
    """Adapter integrating cross-encoder models into the Reranker interface."""

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
            try:
                from FlagEmbedding import FlagReranker
                if not hasattr(self, "_model") or self._model is None:
                    use_fp16 = self.device != "cpu"
                    self._model = FlagReranker(self.model_name_or_path, use_fp16=use_fp16, device=self.device)
                raw = self._model.compute_score(pairs, batch_size=self.batch_size)
                scores = [float(raw)] if isinstance(raw, (int, float)) else [float(s) for s in raw]
            except Exception:
                scores = [
                    float(item.score if item.score is not None else 0.0)
                    for item in selected_candidates
                ]

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
