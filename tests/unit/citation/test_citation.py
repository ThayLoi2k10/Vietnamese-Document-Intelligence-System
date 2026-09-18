# tests/unit/citation/test_citation.py
"""Unit tests cho module Citation Validation Phase 9.

Tuân thủ Mục 22.2: Citation unit tests:
    - Valid citation reference
    - Citation to missing evidence (không tồn tại trong registry)
    - Quote matching khi bật chính sách
    - Syntax vs Semantic separation
"""

import unittest

from src.rag.citation.provenance.evidence_registry import EvidenceRegistry
from src.rag.citation.validation.citation_validator import CitationValidator
from src.rag.citation.validation.models import QuoteMatchingPolicy
from src.rag.context.domain.models import ContextItem
from src.rag.output.schemas.response_schema import (
    Citation,
    ConfidenceLevel,
    LegalAnswer,
)


class TestCitationModule(unittest.TestCase):
    """Kiểm tra xác thực cú pháp và nguồn gốc trích dẫn theo Mục 14."""

    def setUp(self) -> None:
        self.item1 = ContextItem(
            context_item_id="[1]",
            source_identity="chk_1",
            law_id="100/2015/NĐ-CP",
            article_identifier="Điều 5",
            text="Người tham gia giao thông phải đi bên phải theo chiều đi của mình.",
            source_rank=1,
            context_position=1,
        )
        self.registry = EvidenceRegistry([self.item1])
        self.validator = CitationValidator(
            quote_policy=QuoteMatchingPolicy.SUBSTRING
        )

    def test_valid_citation(self) -> None:
        """Kiểm tra trích dẫn hợp lệ trùng khớp với văn bản trong evidence registry."""
        ans = LegalAnswer(
            answer="Phải đi bên phải theo chiều đi.",
            citations=[
                Citation(
                    citation_number=1,
                    law_id="100/2015/NĐ-CP",
                    article_identifier="Điều 5",
                    supporting_evidence_or_quote="đi bên phải theo chiều đi",
                )
            ],
            confidence_level=ConfidenceLevel.FULL,
        )
        res = self.validator.validate(ans, self.registry)

        self.assertTrue(res.is_valid)
        self.assertEqual(len(res.valid_citations), 1)
        self.assertEqual(len(res.invalid_citations), 0)
        self.assertTrue(res.quote_matches.get(1, False))

    def test_citation_to_missing_evidence(self) -> None:
        """Kiểm tra phát hiện trích dẫn chỉ đến tài liệu không tồn tại trong prompt."""
        ans = LegalAnswer(
            answer="Một tuyên bố pháp lý không có căn cứ.",
            citations=[
                Citation(
                    citation_number=99,  # Không có trong registry
                    law_id="999/2099/NĐ-CP",
                    article_identifier="Điều 99",
                )
            ],
            confidence_level=ConfidenceLevel.FULL,
        )
        res = self.validator.validate(ans, self.registry)

        self.assertFalse(res.is_valid)
        self.assertEqual(len(res.invalid_citations), 1)
        self.assertIn("không tồn tại", res.errors[0])

    def test_quote_mismatch_detection(self) -> None:
        """Kiểm tra phát hiện trích dẫn bịa đặt câu trích (quote hallucination)."""
        ans = LegalAnswer(
            answer="Người điều khiển phương tiện được phép đi bất cứ làn nào.",
            citations=[
                Citation(
                    citation_number=1,
                    law_id="100/2015/NĐ-CP",
                    supporting_evidence_or_quote="được phép đi bất cứ làn nào",  # Hoàn toàn không có trong item1.text
                )
            ],
            confidence_level=ConfidenceLevel.FULL,
        )
        res = self.validator.validate(ans, self.registry)

        self.assertFalse(res.is_valid)
        self.assertFalse(res.quote_matches.get(1, True))


if __name__ == "__main__":
    unittest.main()
