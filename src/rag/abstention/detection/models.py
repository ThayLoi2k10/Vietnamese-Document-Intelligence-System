"""Domain models for abstention decisions and failure reason codes."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from src.rag.output.schemas.response_schema import ConfidenceLevel


class AbstentionReasonCode(str, Enum):
    SUFFICIENT_EVIDENCE = "SUFFICIENT_EVIDENCE"
    EMPTY_RETRIEVAL = "EMPTY_RETRIEVAL"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    NO_RELEVANT_LAW = "NO_RELEVANT_LAW"
    MODEL_REPORTED_INSUFFICIENT = "MODEL_REPORTED_INSUFFICIENT"
    MODEL_REPORTED_PARTIAL = "MODEL_REPORTED_PARTIAL"
    INVALID_OR_MISSING_CITATIONS = "INVALID_OR_MISSING_CITATIONS"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


@dataclass
class AbstentionDecision:
    status: ConfidenceLevel
    reason_code: AbstentionReasonCode
    explanation: Optional[str] = None
    supporting_evidence_count: int = 0
    policy_version: str = "abstention-v1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "reason_code": self.reason_code.value,
            "explanation": self.explanation,
            "supporting_evidence_count": self.supporting_evidence_count,
            "policy_version": self.policy_version,
        }
