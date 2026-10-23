"""Domain models for retrieval inputs, items, results, and aggregated articles."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.rag.errors.exceptions import (
    InvalidRetrievalResultError,
    MissingRequiredMetadataError,
)


class RetrievalMode(str, Enum):
    CHUNK = "CHUNK"
    ARTICLE = "ARTICLE"


@dataclass
class RetrievalItem:
    law_id: str
    text: str
    rank: int
    chunk_id: Optional[str] = None
    global_article_id: Optional[int] = None
    article_identifier: Optional[str] = None
    chunk_order: Optional[int] = None
    total_chunks: Optional[int] = None
    score: Optional[float] = None
    token_count: Optional[int] = None
    parent_reference: Optional[str] = None
    source_metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, retrieval_mode: RetrievalMode) -> None:
        if not self.law_id or not str(self.law_id).strip():
            raise MissingRequiredMetadataError(
                "RetrievalItem requires non-empty 'law_id'.",
                details={"item": self.to_dict()},
            )
        if self.text is None:
            raise InvalidRetrievalResultError(
                "RetrievalItem requires non-null 'text'.",
                details={"item": self.to_dict()},
            )
        if self.rank < 1:
            raise InvalidRetrievalResultError(
                f"RetrievalItem rank must be >= 1, got {self.rank}.",
                details={"item": self.to_dict()},
            )

        if retrieval_mode == RetrievalMode.CHUNK:
            if not self.chunk_id or not str(self.chunk_id).strip():
                raise MissingRequiredMetadataError(
                    "CHUNK mode requires non-empty 'chunk_id'.",
                    details={"item": self.to_dict()},
                )
        elif retrieval_mode == RetrievalMode.ARTICLE:
            if self.global_article_id is None and not self.article_identifier:
                raise MissingRequiredMetadataError(
                    "ARTICLE mode requires 'global_article_id' or 'article_identifier'.",
                    details={"item": self.to_dict()},
                )

    @property
    def stable_identifier(self) -> str:
        if self.chunk_id:
            return str(self.chunk_id)
        if self.global_article_id is not None:
            return f"art_{self.global_article_id}"
        return f"{self.law_id}_{self.article_identifier}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArticleResult:
    law_id: str
    article_rank: int
    global_article_id: Optional[int] = None
    article_identifier: Optional[str] = None
    best_chunk_score: Optional[float] = None
    contributing_chunks: List[str] = field(default_factory=list)
    article_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalMetadata:
    retriever_name: str
    retrieval_mode: RetrievalMode
    top_k: int
    component_version: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["retrieval_mode"] = self.retrieval_mode.value
        return d


@dataclass
class RetrievalResult:
    query: str
    retrieval_mode: RetrievalMode
    items: List[RetrievalItem]
    retriever_metadata: RetrievalMetadata
    aggregated_articles: Optional[List[ArticleResult]] = None

    @property
    def is_empty(self) -> bool:
        return len(self.items) == 0

    def validate(self) -> None:
        ranks = [item.rank for item in self.items]
        if ranks != sorted(ranks):
            raise InvalidRetrievalResultError(
                f"Retrieval item ranks must be in ascending order, got: {ranks}"
            )
        for item in self.items:
            item.validate(self.retrieval_mode)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "retrieval_mode": self.retrieval_mode.value,
            "items": [item.to_dict() for item in self.items],
            "aggregated_articles": (
                [art.to_dict() for art in self.aggregated_articles]
                if self.aggregated_articles is not None
                else None
            ),
            "retriever_metadata": self.retriever_metadata.to_dict(),
        }


@dataclass
class RetrievalOptions:
    retrieval_top_k: int = 10
    retrieval_mode: Optional[RetrievalMode] = None
    deterministic: bool = True
    component_version: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
