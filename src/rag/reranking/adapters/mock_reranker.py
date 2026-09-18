# src/rag/reranking/adapters/mock_reranker.py
"""Reranker giả lập phục vụ kiểm thử và kiểm chứng pipeline không cần mô hình thật.

Tuân thủ Mục 9.4: Reranker contract.
"""

from typing import Callable, List, Optional

from src.rag.reranking.domain.models import (
    RankedCandidate,
    RankedResults,
    RerankOptions,
)
from src.rag.reranking.interfaces.reranker import Reranker
from src.rag.retrieval.domain.models import RetrievalItem


class MockReranker(Reranker):
    """Mock Reranker cho phép mô phỏng việc xếp hạng lại ứng viên."""

    def __init__(
        self,
        scoring_fn: Optional[Callable[[str, RetrievalItem], float]] = None,
        reverse_order: bool = False,
        component_version: str = "mock-reranker-1.0",
    ) -> None:
        self.scoring_fn = scoring_fn
        self.reverse_order = reverse_order
        self.component_version = component_version

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalItem],
        options: RerankOptions,
    ) -> RankedResults:
        top_k = options.reranker_top_k
        selected_candidates = candidates[:top_k]

        scored_list = []
        for idx, item in enumerate(selected_candidates, start=1):
            if self.scoring_fn is not None:
                new_score = self.scoring_fn(query, item)
            elif self.reverse_order:
                # Đảo ngược thứ hạng để test rerank làm thay đổi thứ tự
                new_score = float(idx)
            else:
                # Giữ nguyên tương đối hoặc sinh điểm rerank giả định
                base = item.score if item.score is not None else 1.0
                new_score = float(base * 1.5)
            scored_list.append((item, new_score))

        # Sắp xếp giảm dần theo new_score
        scored_list.sort(key=lambda x: x[1], reverse=True)

        ranked_candidates: List[RankedCandidate] = []
        for new_rank, (item, sc) in enumerate(scored_list, start=1):
            ranked_candidates.append(
                RankedCandidate(
                    retrieval_item=item,
                    retrieval_rank=item.rank,
                    rerank_rank=new_rank,
                    retrieval_score=item.score,
                    rerank_score=sc,
                )
            )

        return RankedResults(
            query=query,
            candidates=ranked_candidates,
            reranker_metadata={
                "reranker_name": "mock_reranker",
                "component_version": self.component_version,
                "top_k": top_k,
                "input_count": len(candidates),
                "output_count": len(ranked_candidates),
            },
        )
