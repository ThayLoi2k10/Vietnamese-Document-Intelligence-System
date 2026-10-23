"""Abstract reranker interface for the RAG pipeline."""

from abc import ABC, abstractmethod
from typing import List

from src.rag.reranking.domain.models import RankedResults, RerankOptions
from src.rag.retrieval.domain.models import RetrievalItem


class Reranker(ABC):
    """Abstract interface for candidate rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RetrievalItem],
        options: RerankOptions,
    ) -> RankedResults:
        pass
