# src/rag/retrieval/domain/models.py
"""Mô hình miền (Domain Models) cho hệ thống truy xuất Phase 9 RAG.

Tuân thủ Mục 4: Phase 8 -> Phase 9 Interface, Mục 7: Domain Models, Mục 8: Retriever Interface:
    - 4.2 Required normalized retrieval concepts
    - 4.3 Normalized RetrievalResult schema
    - 4.4 RetrievalItem schema (bảo toàn định danh bất biến)
    - 4.5 ArticleResult schema (tổng hợp cấp điều luật)
    - 4.6 Score semantics & 4.7 Rank semantics
    - 8.4 Empty retrieval là một domain state hợp lệ, không phải exception
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.rag.errors.exceptions import (
    InvalidRetrievalResultError,
    MissingRequiredMetadataError,
)


class RetrievalMode(str, Enum):
    """Chế độ truy xuất của hệ thống."""

    CHUNK = "CHUNK"
    ARTICLE = "ARTICLE"


@dataclass
class RetrievalItem:
    """Biểu diễn một đơn vị bằng chứng pháp lý được truy xuất (Chunk hoặc Article)."""

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
        """Xác thực tính toàn vẹn của RetrievalItem theo Mục 4.4 và 8.5."""
        if not self.law_id or not str(self.law_id).strip():
            raise MissingRequiredMetadataError(
                "RetrievalItem thiếu 'law_id' bắt buộc.",
                details={"item": self.to_dict()},
            )
        if self.text is None:
            raise InvalidRetrievalResultError(
                "RetrievalItem thiếu trường 'text'.",
                details={"item": self.to_dict()},
            )
        if self.rank < 1:
            raise InvalidRetrievalResultError(
                f"RetrievalItem có rank không hợp lệ: {self.rank} (phải >= 1).",
                details={"item": self.to_dict()},
            )

        if retrieval_mode == RetrievalMode.CHUNK:
            if not self.chunk_id or not str(self.chunk_id).strip():
                raise MissingRequiredMetadataError(
                    "Ở chế độ CHUNK, RetrievalItem bắt buộc phải có 'chunk_id'.",
                    details={"item": self.to_dict()},
                )
        elif retrieval_mode == RetrievalMode.ARTICLE:
            if self.global_article_id is None and not self.article_identifier:
                raise MissingRequiredMetadataError(
                    "Ở chế độ ARTICLE, RetrievalItem phải có 'global_article_id' hoặc 'article_identifier'.",
                    details={"item": self.to_dict()},
                )

    @property
    def stable_identifier(self) -> str:
        """Trả về định danh duy nhất ổn định dựa trên retrieval mode."""
        if self.chunk_id:
            return str(self.chunk_id)
        if self.global_article_id is not None:
            return f"art_{self.global_article_id}"
        return f"{self.law_id}_{self.article_identifier}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArticleResult:
    """Biểu diễn kết quả tổng hợp cấp điều luật từ một hoặc nhiều chunk."""

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
    """Metadata nội bộ của quá trình truy xuất."""

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
    """Kết quả chuẩn hóa hoàn chỉnh của một lượt truy xuất."""

    query: str
    retrieval_mode: RetrievalMode
    items: List[RetrievalItem]
    retriever_metadata: RetrievalMetadata
    aggregated_articles: Optional[List[ArticleResult]] = None

    @property
    def is_empty(self) -> bool:
        return len(self.items) == 0

    def validate(self) -> None:
        """Kiểm tra tính toàn vẹn của kết quả truy xuất."""
        ranks = [item.rank for item in self.items]
        if ranks != sorted(ranks):
            raise InvalidRetrievalResultError(
                f"Thứ hạng (ranks) của retrieval items không được sắp xếp tăng dần: {ranks}"
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
    """Tùy chọn cho một lệnh truy xuất."""

    retrieval_top_k: int = 10
    retrieval_mode: Optional[RetrievalMode] = None
    deterministic: bool = True
    component_version: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
