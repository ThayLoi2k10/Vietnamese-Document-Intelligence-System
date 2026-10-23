"""Abstention detector evaluating multi-signal evidence sufficiency and confidence."""

from typing import Optional

from src.rag.abstention.detection.models import (
    AbstentionDecision,
    AbstentionReasonCode,
)
from src.rag.citation.validation.models import CitationValidationResult
from src.rag.context.domain.models import AssembledContext
from src.rag.output.schemas.response_schema import ConfidenceLevel, LegalAnswer


class AbstentionDetector:
    """Evaluates context, generation, and citation signals to determine abstention decisions."""

    def __init__(self, policy_version: str = "abstention-detector-v1.0") -> None:
        self.policy_version = policy_version

    def detect(
        self,
        context: AssembledContext,
        answer: Optional[LegalAnswer],
        citation_validation: Optional[CitationValidationResult],
    ) -> AbstentionDecision:
        evidence_count = len(context.items)

        if evidence_count == 0:
            return AbstentionDecision(
                status=ConfidenceLevel.INSUFFICIENT,
                reason_code=AbstentionReasonCode.EMPTY_RETRIEVAL,
                explanation="Không có bất kỳ tài liệu hoặc điều luật nào được tìm thấy để trả lời câu hỏi.",
                supporting_evidence_count=0,
                policy_version=self.policy_version,
            )

        if answer is None:
            return AbstentionDecision(
                status=ConfidenceLevel.INSUFFICIENT,
                reason_code=AbstentionReasonCode.MODEL_REPORTED_INSUFFICIENT,
                explanation="Mô hình không sinh được câu trả lời pháp lý có cấu trúc.",
                supporting_evidence_count=evidence_count,
                policy_version=self.policy_version,
            )

        if answer.confidence_level == ConfidenceLevel.INSUFFICIENT:
            return AbstentionDecision(
                status=ConfidenceLevel.INSUFFICIENT,
                reason_code=AbstentionReasonCode.MODEL_REPORTED_INSUFFICIENT,
                explanation=answer.abstention_reason
                or "Mô hình xác định tài liệu không đủ để đưa ra câu trả lời chắc chắn.",
                supporting_evidence_count=evidence_count,
                policy_version=self.policy_version,
            )

        if answer.confidence_level == ConfidenceLevel.PARTIAL:
            return AbstentionDecision(
                status=ConfidenceLevel.PARTIAL,
                reason_code=AbstentionReasonCode.MODEL_REPORTED_PARTIAL,
                explanation=answer.abstention_reason
                or "Mô hình xác định chỉ có thể trả lời một phần câu hỏi dựa trên dữ liệu có sẵn.",
                supporting_evidence_count=evidence_count,
                policy_version=self.policy_version,
            )

        if citation_validation is not None:
            if not citation_validation.is_valid:
                return AbstentionDecision(
                    status=ConfidenceLevel.PARTIAL,
                    reason_code=AbstentionReasonCode.INVALID_OR_MISSING_CITATIONS,
                    explanation=f"Phát hiện trích dẫn không hợp lệ: {citation_validation.errors}",
                    supporting_evidence_count=len(citation_validation.valid_citations),
                    policy_version=self.policy_version,
                )
            if len(citation_validation.valid_citations) == 0 and evidence_count > 0:
                return AbstentionDecision(
                    status=ConfidenceLevel.PARTIAL,
                    reason_code=AbstentionReasonCode.INVALID_OR_MISSING_CITATIONS,
                    explanation="Câu trả lời không viện dẫn bất kỳ trích dẫn pháp lý cụ thể nào từ ngữ cảnh.",
                    supporting_evidence_count=0,
                    policy_version=self.policy_version,
                )

        return AbstentionDecision(
            status=ConfidenceLevel.FULL,
            reason_code=AbstentionReasonCode.SUFFICIENT_EVIDENCE,
            explanation="Câu trả lời được hỗ trợ đầy đủ bởi các căn cứ và trích dẫn hợp lệ.",
            supporting_evidence_count=(
                len(citation_validation.valid_citations)
                if citation_validation
                else len(answer.citations)
            ),
            policy_version=self.policy_version,
        )
