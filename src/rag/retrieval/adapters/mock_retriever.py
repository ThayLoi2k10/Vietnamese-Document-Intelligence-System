"""Mock retriever implementation for deterministic testing."""

from typing import List, Optional

from src.rag.retrieval.domain.models import (
    RetrievalItem,
    RetrievalMetadata,
    RetrievalMode,
    RetrievalOptions,
    RetrievalResult,
)
from src.rag.retrieval.interfaces.retriever import Retriever


class MockRetriever(Retriever):
    """Configurable mock retriever returning canned items."""

    def __init__(
        self,
        canned_items: Optional[List[RetrievalItem]] = None,
        retrieval_mode: RetrievalMode = RetrievalMode.CHUNK,
        retriever_name: str = "mock_retriever",
        component_version: str = "mock-1.0",
    ) -> None:
        self.canned_items = canned_items or []
        self.retrieval_mode = retrieval_mode
        self.retriever_name = retriever_name
        self.component_version = component_version
        self.last_query: Optional[str] = None
        self.last_options: Optional[RetrievalOptions] = None

    def retrieve(self, query: str, options: RetrievalOptions) -> RetrievalResult:
        self.last_query = query
        self.last_options = options

        top_k = options.retrieval_top_k
        selected_items = self.canned_items[:top_k]

        ranked_items: List[RetrievalItem] = []
        for idx, it in enumerate(selected_items, start=1):
            ranked_items.append(
                RetrievalItem(
                    chunk_id=it.chunk_id,
                    global_article_id=it.global_article_id,
                    law_id=it.law_id,
                    article_identifier=it.article_identifier,
                    text=it.text,
                    chunk_order=it.chunk_order,
                    total_chunks=it.total_chunks,
                    score=it.score,
                    rank=idx,
                    token_count=it.token_count,
                    parent_reference=it.parent_reference,
                    source_metadata=dict(it.source_metadata),
                )
            )

        metadata = RetrievalMetadata(
            retriever_name=self.retriever_name,
            retrieval_mode=self.retrieval_mode,
            top_k=top_k,
            component_version=self.component_version,
            parameters=options.parameters,
        )

        res = RetrievalResult(
            query=query,
            retrieval_mode=self.retrieval_mode,
            items=ranked_items,
            retriever_metadata=metadata,
        )
        res.validate()
        return res
