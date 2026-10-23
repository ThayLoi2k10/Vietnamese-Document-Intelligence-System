"""Domain models for unified RAG pipeline responses and audit traces."""

from dataclasses import dataclass, field
from typing import Any, Dict

from src.rag.abstention.detection.models import AbstentionDecision
from src.rag.citation.validation.models import CitationValidationResult
from src.rag.context.domain.models import AssembledContext
from src.rag.generation.domain.models import GenerationResult
from src.rag.output.schemas.response_schema import LegalAnswer


@dataclass
class RAGResponse:
    question: str
    legal_answer: LegalAnswer
    citation_validation: CitationValidationResult
    abstention_decision: AbstentionDecision
    assembled_context: AssembledContext
    generation_result: GenerationResult
    execution_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> Dict[str, Any]:
        return self.legal_answer.to_dict()

    def to_full_audit_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "legal_answer": self.legal_answer.to_dict(),
            "citation_validation": self.citation_validation.to_dict(),
            "abstention_decision": self.abstention_decision.to_dict(),
            "assembled_context": self.assembled_context.to_dict(),
            "generation_result": self.generation_result.to_dict(),
            "execution_metadata": self.execution_metadata,
        }
