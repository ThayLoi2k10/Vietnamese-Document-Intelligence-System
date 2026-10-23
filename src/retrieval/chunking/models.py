"""Mô hình dữ liệu LegalChunk và các hàm băm định danh tất định cho phân mảnh văn bản."""

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Dict, Optional


def compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_deterministic_chunk_id(
    global_article_id: int,
    chunking_version: str,
    chunk_order: int,
    content_hash: str,
) -> str:
    short_hash = content_hash[:8] if content_hash else "empty000"
    return f"{global_article_id}_{chunking_version}_{chunk_order:03d}_{short_hash}"


@dataclass
class LegalChunk:
    chunk_id: str
    global_article_id: int
    law_id: str
    article_identifier: str
    chunk_order: int
    total_chunks: int
    chunk_text: str
    token_count: int
    structure_path: str
    content_hash: str
    chunking_strategy: str
    chunking_version: str
    char_start: int
    char_end: int
    parent_section_id: Optional[str] = None
    parent_clause_id: Optional[str] = None
    parent_point_id: Optional[str] = None
    structural_unit: Optional[str] = None
    structural_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LegalChunk":
        return cls(
            chunk_id=str(data["chunk_id"]),
            global_article_id=int(data["global_article_id"]),
            law_id=str(data["law_id"]),
            article_identifier=str(data["article_identifier"]),
            chunk_order=int(data["chunk_order"]),
            total_chunks=int(data["total_chunks"]),
            chunk_text=str(data["chunk_text"]),
            token_count=int(data["token_count"]),
            structure_path=str(data["structure_path"]),
            content_hash=str(data["content_hash"]),
            chunking_strategy=str(data["chunking_strategy"]),
            chunking_version=str(data["chunking_version"]),
            char_start=int(data["char_start"]),
            char_end=int(data["char_end"]),
            parent_section_id=data.get("parent_section_id"),
            parent_clause_id=data.get("parent_clause_id"),
            parent_point_id=data.get("parent_point_id"),
            structural_unit=data.get("structural_unit"),
            structural_label=data.get("structural_label"),
        )
