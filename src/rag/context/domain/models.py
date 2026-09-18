# src/rag/context/domain/models.py
"""Mô hình miền cho hệ thống Context Assembly Phase 9 RAG.

Tuân thủ:
    - Mục 7.3: Provenance model
    - Mục 7.4: Internal vs public metadata (Ranh giới bảo vệ metadata nội bộ)
    - Mục 10.7: Context representation
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ContextItem:
    """Đại diện một đơn vị ngữ cảnh được tuyển chọn đưa vào prompt context."""

    context_item_id: str  # Ví dụ: "[1]", "[2]"
    source_identity: str
    law_id: str
    text: str
    source_rank: int
    context_position: int  # 1-indexed trong AssembledContext
    article_identifier: Optional[str] = None
    global_article_id: Optional[int] = None
    chunk_id: Optional[str] = None
    source_score: Optional[float] = None
    token_count: Optional[int] = None
    internal_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_llm_visible_dict(self) -> Dict[str, Any]:
        """Chỉ xuất các trường thông tin cần thiết cho LLM theo Mục 7.4.

        Loại bỏ tuyệt đối: vector score, internal rank, tên adapter, database IDs.
        """
        data: Dict[str, Any] = {
            "reference_id": self.context_item_id,
            "law_id": self.law_id,
            "text": self.text,
        }
        if self.article_identifier:
            data["article_identifier"] = self.article_identifier
        return data

    def to_provenance_dict(self) -> Dict[str, Any]:
        """Xuất đầy đủ dữ liệu provenance để phục vụ kiểm toán và xác minh trích dẫn."""
        return asdict(self)


@dataclass
class AssembledContext:
    """Tập ngữ cảnh hoàn chỉnh sau khi lọc, khử trùng, mở rộng, sắp xếp và kiểm tra budget."""

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
