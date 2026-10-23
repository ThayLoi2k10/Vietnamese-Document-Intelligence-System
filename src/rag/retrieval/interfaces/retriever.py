"""Abstract retriever interface for the RAG pipeline."""

from abc import ABC, abstractmethod
from typing import List

from src.rag.retrieval.domain.models import RetrievalOptions, RetrievalResult


class Retriever(ABC):
    """Abstract interface for document and chunk retrieval components."""

    @abstractmethod
    def retrieve(self, query: str, options: RetrievalOptions) -> RetrievalResult:
        pass

    def batch_retrieve(
        self, queries: List[str], options: RetrievalOptions
    ) -> List[RetrievalResult]:
        return [self.retrieve(q, options) for q in queries]
