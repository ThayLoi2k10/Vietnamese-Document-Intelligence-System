# src/rag/output/schemas/response_schema.py
"""Schema công khai cho câu trả lời pháp lý có cấu trúc (Structured Legal Answer).

Tuân thủ Mục 13: Structured Output:
    - 13.1 Public schema: answer, citations, confidence_level, abstention_reason
    - 13.2 Citation object
    - 13.3 Confidence semantics (FULL, PARTIAL, INSUFFICIENT)
    - 13.4 Không chứa reasoning trace trong schema công khai
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.rag.errors.exceptions import MalformedStructuredOutputError


class ConfidenceLevel(str, Enum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass
class Citation:
    """Đại diện một trích dẫn pháp lý trong câu trả lời."""

    citation_number: int
    law_id: str
    article_identifier: Optional[str] = None
    chunk_id: Optional[str] = None
    global_article_id: Optional[int] = None
    supporting_evidence_or_quote: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Citation":
        if "citation_number" not in data or "law_id" not in data:
            raise MalformedStructuredOutputError(
                "Citation thiếu trường bắt buộc 'citation_number' hoặc 'law_id'.",
                details={"data": data},
            )
        try:
            c_num = int(data["citation_number"])
        except (ValueError, TypeError):
            raise MalformedStructuredOutputError(
                f"Trường 'citation_number' phải là số nguyên: {data.get('citation_number')}"
            )

        return cls(
            citation_number=c_num,
            law_id=str(data["law_id"]),
            article_identifier=(
                str(data["article_identifier"])
                if data.get("article_identifier") is not None
                else None
            ),
            chunk_id=(
                str(data["chunk_id"]) if data.get("chunk_id") is not None else None
            ),
            global_article_id=(
                int(data["global_article_id"])
                if data.get("global_article_id") is not None
                else None
            ),
            supporting_evidence_or_quote=(
                str(data["supporting_evidence_or_quote"])
                if data.get("supporting_evidence_or_quote") is not None
                else None
            ),
        )


@dataclass
class LegalAnswer:
    """Cấu trúc công khai của một câu trả lời pháp lý chuẩn."""

    answer: str
    citations: List[Citation]
    confidence_level: ConfidenceLevel
    abstention_reason: Optional[str] = None

    def validate(self) -> None:
        """Kiểm tra tính hợp lệ của câu trả lời."""
        if self.answer is None:
            raise MalformedStructuredOutputError("Trường 'answer' không được là None.")
        if not isinstance(self.citations, list):
            raise MalformedStructuredOutputError("Trường 'citations' phải là một danh sách.")
        if not isinstance(self.confidence_level, ConfidenceLevel):
            raise MalformedStructuredOutputError(
                f"Trường 'confidence_level' không hợp lệ: {self.confidence_level}"
            )
        if self.confidence_level == ConfidenceLevel.INSUFFICIENT and not self.abstention_reason:
            # Khuyến nghị hoặc yêu cầu lý do từ chối khi INSUFFICIENT
            pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "confidence_level": self.confidence_level.value,
            "abstention_reason": self.abstention_reason,
        }
