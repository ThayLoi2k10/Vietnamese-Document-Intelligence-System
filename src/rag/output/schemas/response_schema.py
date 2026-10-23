"""Public schema for structured legal answers and citations."""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from src.rag.errors.exceptions import MalformedStructuredOutputError


class ConfidenceLevel(str, Enum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass
class Citation:
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
                "Citation requires 'citation_number' and 'law_id'.",
                details={"data": data},
            )
        try:
            c_num = int(data["citation_number"])
        except (ValueError, TypeError):
            raise MalformedStructuredOutputError(
                f"Field 'citation_number' must be an integer, got: {data.get('citation_number')}"
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
    answer: str
    citations: List[Citation]
    confidence_level: ConfidenceLevel
    abstention_reason: Optional[str] = None

    def validate(self) -> None:
        if self.answer is None:
            raise MalformedStructuredOutputError("Field 'answer' cannot be None.")
        if not isinstance(self.citations, list):
            raise MalformedStructuredOutputError("Field 'citations' must be a list.")
        if not isinstance(self.confidence_level, ConfidenceLevel):
            raise MalformedStructuredOutputError(
                f"Invalid 'confidence_level': {self.confidence_level}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "confidence_level": self.confidence_level.value,
            "abstention_reason": self.abstention_reason,
        }
