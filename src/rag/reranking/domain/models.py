"""Domain models for reranked candidates and ranking options."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.rag.retrieval.domain.models import RetrievalItem


@dataclass
class RankedCandidate:
    retrieval_item: RetrievalItem
    retrieval_rank: int
    rerank_rank: int
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieval_item": self.retrieval_item.to_dict(),
            "retrieval_rank": self.retrieval_rank,
            "rerank_rank": self.rerank_rank,
            "retrieval_score": self.retrieval_score,
            "rerank_score": self.rerank_score,
        }


@dataclass
class RankedResults:
    query: str
    candidates: List[RankedCandidate]
    reranker_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "candidates": [c.to_dict() for c in self.candidates],
            "reranker_metadata": self.reranker_metadata,
        }


@dataclass
class RerankOptions:
    reranker_top_k: int = 10
    component_version: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
