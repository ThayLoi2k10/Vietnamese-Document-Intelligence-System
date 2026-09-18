# src/retrieval/indexing/models.py
"""Mô hình dữ liệu điểm vector và kết quả truy vấn cho Phase 8.

Tuân thủ nghiêm ngặt:
    - Mục 11: Vector Point Schema (Identity, Provenance, Retrieval Metadata, Payload text)
    - Mục 12: Deterministic Point Identity (UUID v5 từ chunk_id)
    - Mục 24: Query Interface Contract (VectorSearchResult tương thích với Phase 7 aggregation)
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import uuid

# Namespace UUID cố định cho định danh điểm tất định trong dự án
LEGAL_VECTOR_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def generate_deterministic_point_id(chunk_id: str) -> str:
    """Sinh point_id tất định định dạng UUID từ chunk_id theo Mục 12."""
    return str(uuid.uuid5(LEGAL_VECTOR_NAMESPACE, str(chunk_id)))


@dataclass
class VectorPoint:
    """Đại diện một điểm vector hoàn chỉnh trong cơ sở dữ liệu vector."""

    point_id: str
    chunk_id: str
    vector: List[float]
    global_article_id: int
    law_id: str
    article_identifier: str
    chunk_order: int
    total_chunks: int
    chunking_version: str
    token_count: int
    content_hash: str
    chunking_strategy: str
    chunk_text: str
    is_active: bool = True
    char_start: int = 0
    char_end: int = 0
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """Trích xuất payload metadata lưu trong vector store."""
        return {
            "chunk_id": self.chunk_id,
            "global_article_id": int(self.global_article_id),
            "law_id": str(self.law_id),
            "article_identifier": str(self.article_identifier),
            "chunk_order": int(self.chunk_order),
            "total_chunks": int(self.total_chunks),
            "chunking_version": str(self.chunking_version),
            "token_count": int(self.token_count),
            "content_hash": str(self.content_hash),
            "chunking_strategy": str(self.chunking_strategy),
            "chunk_text": str(self.chunk_text),
            "is_active": bool(self.is_active),
            "char_start": int(self.char_start),
            "char_end": int(self.char_end),
            **self.extra_metadata,
        }

    @classmethod
    def from_chunk_and_vector(
        cls,
        chunk_dict: Dict[str, Any],
        vector: List[float],
        is_active: bool = True,
    ) -> "VectorPoint":
        """Khởi tạo VectorPoint từ từ điển chunk và vector BGE-M3."""
        cid = str(chunk_dict["chunk_id"])
        pid = generate_deterministic_point_id(cid)
        return cls(
            point_id=pid,
            chunk_id=cid,
            vector=vector,
            global_article_id=int(chunk_dict["global_article_id"]),
            law_id=str(chunk_dict["law_id"]),
            article_identifier=str(chunk_dict["article_identifier"]),
            chunk_order=int(chunk_dict["chunk_order"]),
            total_chunks=int(chunk_dict["total_chunks"]),
            chunking_version=str(chunk_dict["chunking_version"]),
            token_count=int(chunk_dict["token_count"]),
            content_hash=str(chunk_dict["content_hash"]),
            chunking_strategy=str(chunk_dict["chunking_strategy"]),
            chunk_text=str(chunk_dict["chunk_text"]),
            is_active=is_active,
            char_start=int(chunk_dict.get("char_start", 0)),
            char_end=int(chunk_dict.get("char_end", 0)),
        )


@dataclass
class VectorSearchResult:
    """Kết quả trả về từ tìm kiếm tương đồng vector."""

    point_id: str
    chunk_id: str
    global_article_id: int
    score: float
    rank: int
    payload: Dict[str, Any] = field(default_factory=dict)
