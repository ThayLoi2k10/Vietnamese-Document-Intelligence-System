# tests/unit/abstention/test_abstention.py
"""Unit tests cho module Abstention Phase 9.

Tuân thủ Mục 22.2: Abstention unit tests:
    - Sufficient context + answer -> FULL
    - Insufficient retrieval -> INSUFFICIENT / EMPTY_RETRIEVAL
    - Explicit structured abstention
    - Kháng cự từ khóa ngây thơ (không phụ thuộc keyword 'không biết')
"""

import unittest

from src.rag.abstention.detection.abstention_detector import AbstentionDetector
from src.rag.abstention.detection.models import AbstentionReasonCode
from src.rag.citation.validation.models import CitationValidationResult
from src.rag.context.domain.models import AssembledContext
from src.rag.output.schemas.response_schema import (
    Citation,
    ConfidenceLevel,
    LegalAnswer,
)
from tests.fixtures.sample_retrieval import make_sample_context_items


class TestAbstentionModule(unittest.TestCase):
    """Kiểm tra cơ chế ra quyết định từ chối trả lời theo Mục 15."""

    def setUp(self) -> None:
        self.detector = AbstentionDetector()
        self.context_items = make_sample_context_items()
        self.context = AssembledContext(
            query="câu hỏi",
            items=self.context_items,
            total_tokens=50,
        )

    def test_sufficient_context_and_answer(self) -> None:
        """Kiểm tra trường hợp đủ căn cứ và trích dẫn hợp lệ -> FULL."""
        ans = LegalAnswer(
            answer="Phải chấp hành hiệu lệnh của cảnh sát giao thông.",
            citations=[
                Citation(
                    citation_number=1,
                    law_id="100/2015/NĐ-CP",
                    article_identifier="Điều 5",
                )
            ],
            confidence_level=ConfidenceLevel.FULL,
        )
        val_res = CitationValidationResult(
            is_valid=True,
            valid_citations=ans.citations,
        )

        decision = self.detector.detect(self.context, ans, val_res)

        self.assertEqual(decision.status, ConfidenceLevel.FULL)
        self.assertEqual(
            decision.reason_code, AbstentionReasonCode.SUFFICIENT_EVIDENCE
        )

    def test_empty_retrieval_leads_to_insufficient(self) -> None:
        """Kiểm tra trường hợp truy xuất không ra tài liệu nào -> INSUFFICIENT."""
        empty_context = AssembledContext(
            query="câu hỏi về luật không có thật",
            items=[],
            total_tokens=0,
        )
        decision = self.detector.detect(empty_context, None, None)

        self.assertEqual(decision.status, ConfidenceLevel.INSUFFICIENT)
        self.assertEqual(
            decision.reason_code, AbstentionReasonCode.EMPTY_RETRIEVAL
        )

    def test_model_reported_insufficient(self) -> None:
        """Kiểm tra mô hình chủ động báo INSUFFICIENT qua schema có cấu trúc."""
        ans = LegalAnswer(
            answer="Tài liệu không có quy định về câu hỏi này.",
            citations=[],
            confidence_level=ConfidenceLevel.INSUFFICIENT,
            abstention_reason="Không tìm thấy điều luật áp dụng",
        )
        decision = self.detector.detect(self.context, ans, None)

        self.assertEqual(decision.status, ConfidenceLevel.INSUFFICIENT)
        self.assertEqual(
            decision.reason_code,
            AbstentionReasonCode.MODEL_REPORTED_INSUFFICIENT,
        )
        self.assertEqual(
            decision.explanation, "Không tìm thấy điều luật áp dụng"
        )

    def test_keyword_immunity(self) -> None:
        """Kiểm tra câu trả lời có từ 'không biết' trong văn cảnh nhưng đủ trích dẫn thì KHÔNG bị ép từ chối (Mục 15.4)."""
        # Ví dụ câu hỏi: Người tham gia giao thông có quyền nói 'tôi không biết luật' để tránh phạt không?
        ans = LegalAnswer(
            answer="Người vi phạm không được lấy lý do 'không biết' để từ chối chấp hành việc xử phạt vi phạm hành chính.",
            citations=[
                Citation(
                    citation_number=1,
                    law_id="100/2015/NĐ-CP",
                    article_identifier="Điều 5",
                )
            ],
            confidence_level=ConfidenceLevel.FULL,
        )
        val_res = CitationValidationResult(
            is_valid=True,
            valid_citations=ans.citations,
        )

        decision = self.detector.detect(self.context, ans, val_res)
        # Quyết định vẫn là FULL chứ không bị ép thành INSUFFICIENT chỉ vì có chữ "không biết"
        self.assertEqual(decision.status, ConfidenceLevel.FULL)


if __name__ == "__main__":
    unittest.main()
