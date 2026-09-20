# tests/unit/output/test_output.py
"""Unit tests cho module Structured Output Parsing Phase 9.

Tuân thủ Mục 22.2: Output unit tests:
    - Valid structured output parsing (cả có và không có markdown code fence)
    - Malformed JSON/schema rejection
    - Enum validation for confidence state
    - Abstention-reason validation
    - Loại bỏ và không chấp nhận reasoning traces trong schema
"""

import unittest

from src.rag.errors.exceptions import MalformedStructuredOutputError
from src.rag.output.parsers.structured_parser import OutputParser
from src.rag.output.schemas.response_schema import ConfidenceLevel


class TestOutputModule(unittest.TestCase):
    """Kiểm tra tính nghiêm ngặt của OutputParser theo Mục 13."""

    def setUp(self) -> None:
        self.parser = OutputParser()

    def test_parse_valid_json_output(self) -> None:
        """Kiểm tra parse thành công JSON hợp lệ đầy đủ các trường."""
        raw = """{
            "answer": "Người điều khiển xe phải chấp hành hiệu lệnh của cảnh sát giao thông.",
            "citations": [
                {
                    "citation_number": 1,
                    "law_id": "100/2015/NĐ-CP",
                    "article_identifier": "Điều 5",
                    "supporting_evidence_or_quote": "phải chấp hành hiệu lệnh"
                }
            ],
            "confidence_level": "FULL",
            "abstention_reason": null
        }"""
        answer = self.parser.parse(raw)

        self.assertEqual(answer.confidence_level, ConfidenceLevel.FULL)
        self.assertEqual(len(answer.citations), 1)
        self.assertEqual(answer.citations[0].citation_number, 1)
        self.assertEqual(answer.citations[0].law_id, "100/2015/NĐ-CP")
        self.assertIsNone(answer.abstention_reason)

    def test_parse_json_wrapped_in_markdown_code_block(self) -> None:
        """Kiểm tra bóc tách thành công JSON được bọc bởi ```json ... ```."""
        raw = """```json
        {
            "answer": "Quy định về xử phạt hành chính.",
            "citations": [],
            "confidence_level": "PARTIAL",
            "abstention_reason": "Chưa rõ mức phạt cụ thể"
        }
        ```"""
        answer = self.parser.parse(raw)

        self.assertEqual(answer.confidence_level, ConfidenceLevel.PARTIAL)
        self.assertEqual(answer.abstention_reason, "Chưa rõ mức phạt cụ thể")

    def test_reject_malformed_json_syntax(self) -> None:
        """Kiểm tra ném MalformedStructuredOutputError khi cú pháp JSON lỗi."""
        raw = "{ 'answer': không có ngoặc kép đúng cú pháp JSON "
        with self.assertRaises(MalformedStructuredOutputError):
            self.parser.parse(raw)

    def test_reject_invalid_confidence_enum(self) -> None:
        """Kiểm tra ném lỗi khi confidence_level không thuộc [FULL, PARTIAL, INSUFFICIENT]."""
        raw = """{
            "answer": "Câu trả lời...",
            "citations": [],
            "confidence_level": "VERY_HIGH_CONFIDENCE",
            "abstention_reason": null
        }"""
        with self.assertRaises(MalformedStructuredOutputError):
            self.parser.parse(raw)

    def test_reject_missing_required_fields(self) -> None:
        """Kiểm tra ném lỗi khi thiếu trường answer hoặc citations."""
        # Thiếu citations
        raw = """{
            "answer": "Câu trả lời...",
            "confidence_level": "FULL"
        }"""
        with self.assertRaises(MalformedStructuredOutputError) as ctx:
            self.parser.parse(raw)
        self.assertIn("raw_text", ctx.exception.details)

    def test_reject_empty_output(self) -> None:
        """Kiểm tra ném lỗi khi đầu ra rỗng hoặc chỉ có khoảng trắng / rỗng sau code fence."""
        # 1. Chuỗi rỗng hoàn toàn
        with self.assertRaises(MalformedStructuredOutputError) as ctx1:
            self.parser.parse("")
        self.assertEqual(ctx1.exception.details.get("raw_text"), "")

        # 2. Chuỗi chỉ có khoảng trắng
        with self.assertRaises(MalformedStructuredOutputError) as ctx2:
            self.parser.parse("   \n\t  ")
        self.assertEqual(ctx2.exception.details.get("raw_text"), "   \n\t  ")

        # 3. Khối markdown code fence rỗng
        raw_empty_fence = "```json\n```"
        with self.assertRaises(MalformedStructuredOutputError) as ctx3:
            self.parser.parse(raw_empty_fence)
        self.assertEqual(ctx3.exception.details.get("raw_text"), raw_empty_fence)
        self.assertIn("raw_text", ctx3.exception.details)

    def test_reject_truncated_json(self) -> None:
        """Kiểm tra ném lỗi khi đầu ra JSON bị cắt cụt (truncated JSON)."""
        raw_truncated = '{"answer": "Người tham gia giao thông...", "citations": ['
        with self.assertRaises(MalformedStructuredOutputError) as ctx:
            self.parser.parse(raw_truncated)
        self.assertEqual(ctx.exception.details.get("raw_text"), raw_truncated)
        self.assertIn("json_str", ctx.exception.details)

    def test_reject_non_dict_json(self) -> None:
        """Kiểm tra ném lỗi khi đầu ra JSON không phải dict (ví dụ list hoặc scalar)."""
        raw_list = '["answer", "citations", "confidence_level"]'
        with self.assertRaises(MalformedStructuredOutputError) as ctx:
            self.parser.parse(raw_list)
        self.assertEqual(ctx.exception.details.get("raw_text"), raw_list)
        self.assertIn("data", ctx.exception.details)


if __name__ == "__main__":
    unittest.main()
