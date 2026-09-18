# src/rag/citation/validation/models.py
"""Mô hình miền cho xác thực trích dẫn (Citation Validation).

Tuân thủ Mục 14: Citation Validation:
    - 14.2 Minimum syntactic/provenance checks
    - 14.3 Tách bạch Syntactic validity vs Semantic correctness
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List

from src.rag.output.schemas.response_schema import Citation


class QuoteMatchingPolicy(str, Enum):
    """Chính sách kiểm tra đoạn trích dẫn (quote matching) so với văn bản gốc."""

    DISABLED = "DISABLED"
    EXACT = "EXACT"
    SUBSTRING = "SUBSTRING"
    NORMALIZED_FUZZY = "NORMALIZED_FUZZY"


@dataclass
class CitationValidationResult:
    """Kết quả xác thực tính hợp lệ về mặt cú pháp và nguồn gốc của các trích dẫn."""

    is_valid: bool
    valid_citations: List[Citation] = field(default_factory=list)
    invalid_citations: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    quote_matches: Dict[int, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "valid_count": len(self.valid_citations),
            "invalid_count": len(self.invalid_citations),
            "errors": self.errors,
            "invalid_citations": self.invalid_citations,
            "quote_matches": self.quote_matches,
        }
