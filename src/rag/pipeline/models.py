# src/rag/pipeline/models.py
"""Mô hình miền đại diện cho kết quả hoàn chỉnh của Pipeline RAG.

Tuân thủ Mục 7.1: Core value objects & Mục 16: RAG Pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from src.rag.abstention.detection.models import AbstentionDecision
from src.rag.citation.validation.models import CitationValidationResult
from src.rag.context.domain.models import AssembledContext
from src.rag.generation.domain.models import GenerationResult
from src.rag.output.schemas.response_schema import LegalAnswer


@dataclass
class RAGResponse:
    """Phản hồi hoàn chỉnh của pipeline RAG chứa cả kết quả công khai lẫn vết kiểm toán nội bộ."""

    question: str
    legal_answer: LegalAnswer
    citation_validation: CitationValidationResult
    abstention_decision: AbstentionDecision
    assembled_context: AssembledContext
    generation_result: GenerationResult
    execution_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> Dict[str, Any]:
        """Xuất phản hồi công khai tuân thủ nghiêm ngặt Mục 13.1 (Public Schema).

        Không chứa bất kỳ reasoning trace hoặc metadata nội bộ.
        """
        return self.legal_answer.to_dict()

    def to_full_audit_dict(self) -> Dict[str, Any]:
        """Xuất toàn bộ thông tin phục vụ phân tích lỗi, đánh giá và tái lập kết quả."""
        return {
            "question": self.question,
            "legal_answer": self.legal_answer.to_dict(),
            "citation_validation": self.citation_validation.to_dict(),
            "abstention_decision": self.abstention_decision.to_dict(),
            "assembled_context": self.assembled_context.to_dict(),
            "generation_result": self.generation_result.to_dict(),
            "execution_metadata": self.execution_metadata,
        }
