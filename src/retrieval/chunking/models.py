# src/retrieval/chunking/models.py
"""Mô hình dữ liệu và định danh chunk văn bản pháp luật cho Phase 7.

Tuân thủ nghiêm ngặt đặc tả Phase 7:
    - Mục 11: Chunk Data Model (tối thiểu 12 trường cốt lõi + trường cấu trúc)
    - Mục 12: Provenance Invariants (bảo toàn nguồn gốc điều luật, 8 điều luật rỗng)
    - Mục 13: Chunk Identity and Versioning (định danh tất định)
"""

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Dict, List, Optional


def compute_content_hash(text: str) -> str:
    """Tính mã SHA-256 băm nội dung chunk văn bản."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_deterministic_chunk_id(
    global_article_id: int,
    chunking_version: str,
    chunk_order: int,
    content_hash: str,
) -> str:
    """Sinh định danh tất định (deterministic chunk_id) theo đặc tả Mục 13.
    
    Cấu trúc: {global_article_id}_{chunking_version}_{chunk_order}_{content_hash[:8]}
    """
    short_hash = content_hash[:8] if content_hash else "empty000"
    return f"{global_article_id}_{chunking_version}_{chunk_order:03d}_{short_hash}"


@dataclass
class LegalChunk:
    """Mô hình dữ liệu đại diện cho một chunk điều luật pháp lý."""

    # Định danh & nguồn gốc cốt lõi
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

    # Vị trí ký tự trong điều luật nguồn (12 trường bắt buộc)
    char_start: int
    char_end: int

    # Metadata phân cấp cấu trúc pháp lý (tùy chọn)
    parent_section_id: Optional[str] = None
    parent_clause_id: Optional[str] = None
    parent_point_id: Optional[str] = None
    structural_unit: Optional[str] = None
    structural_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành từ điển phục vụ lưu trữ Parquet / JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LegalChunk":
        """Khởi tạo LegalChunk từ từ điển dữ liệu."""
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
