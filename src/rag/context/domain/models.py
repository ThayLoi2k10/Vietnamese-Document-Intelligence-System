"""Domain models for assembled context and evidence items."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ContextItem:
    context_item_id: str
    source_identity: str
    law_id: str
    text: str
    source_rank: int
    context_position: int
    article_identifier: Optional[str] = None
    global_article_id: Optional[int] = None
    chunk_id: Optional[str] = None
    source_score: Optional[float] = None
    token_count: Optional[int] = None
    internal_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_llm_visible_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "reference_id": self.context_item_id,
            "law_id": self.law_id,
            "text": self.text,
        }
        if self.article_identifier:
            data["article_identifier"] = self.article_identifier
        return data

    def to_provenance_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AssembledContext:
    query: str
    items: List[ContextItem]
    total_tokens: int
    evidence_registry: Dict[str, ContextItem] = field(default_factory=dict)
    context_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return len(self.items) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "items": [item.to_provenance_dict() for item in self.items],
            "total_tokens": self.total_tokens,
            "context_metadata": self.context_metadata,
        }
