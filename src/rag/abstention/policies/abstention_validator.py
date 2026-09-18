# src/rag/abstention/policies/abstention_validator.py
"""Bộ kiểm định tính nhất quán của quyết định từ chối (AbstentionValidator).

Tuân thủ Mục 15.6: Abstention evaluation & Validation.
"""

from typing import Optional

from src.rag.abstention.detection.models import (
    AbstentionDecision,
    AbstentionReasonCode,
)
from src.rag.abstention.policies.abstention_policy import AbstentionPolicy
from src.rag.citation.validation.models import CitationValidationResult
from src.rag.context.domain.models import AssembledContext
from src.rag.output.schemas.response_schema import ConfidenceLevel, LegalAnswer


class AbstentionValidator:
    """Xác thực tính hợp lệ và nhất quán của quyết định từ chối so với câu trả lời."""

    def __init__(self, policy: Optional[AbstentionPolicy] = None) -> None:
        self.policy = policy or AbstentionPolicy()

    def validate_and_reconcile(
        self,
        decision: AbstentionDecision,
        answer: LegalAnswer,
        context: AssembledContext,
        citation_validation: Optional[CitationValidationResult] = None,
    ) -> AbstentionDecision:
        """Kiểm tra và điều hòa (reconcile) quyết định từ chối trả lời."""
        # Nếu context hoàn toàn không có bằng chứng
        if len(context.items) == 0:
            return AbstentionDecision(
                status=ConfidenceLevel.INSUFFICIENT,
                reason_code=AbstentionReasonCode.EMPTY_RETRIEVAL,
                explanation="Không có tài liệu nào trong ngữ cảnh.",
                supporting_evidence_count=0,
                policy_version=self.policy.policy_version,
            )

        # Nếu có trích dẫn bị lỗi và chính sách yêu cầu hạ mức tin cậy
        if self.policy.downgrade_on_invalid_citations:
            if citation_validation and not citation_validation.is_valid:
                return AbstentionDecision(
                    status=ConfidenceLevel.PARTIAL,
                    reason_code=AbstentionReasonCode.INVALID_OR_MISSING_CITATIONS,
                    explanation="Trích dẫn không hợp lệ, hạ mức tin cậy xuống PARTIAL.",
                    supporting_evidence_count=len(citation_validation.valid_citations),
                    policy_version=self.policy.policy_version,
                )

        return decision
