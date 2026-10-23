"""Unit tests for offline legal query normalization and expansion pipeline."""

import json
import unittest

from src.rag.query.models import FailureType, ProcessingStatus
from src.rag.query.normalizer import LegalAcronymNormalizer, NormalizationMode
from src.rag.query.processor import QueryProcessor
from src.rag.query.rewriter import LLMQueryRewriter


class TestLegalAcronymNormalizer(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer_expand = LegalAcronymNormalizer(mode=NormalizationMode.EXPAND)
        self.normalizer_replace = LegalAcronymNormalizer(mode=NormalizationMode.REPLACE)
        self.normalizer_dual = LegalAcronymNormalizer(mode=NormalizationMode.DUAL_SPACE)

    def test_basic_acronym_expand_mode(self) -> None:
        query = "Quyền lợi của NLĐ khi giao kết HĐLĐ?"
        norm, acrs, rules = self.normalizer_expand.normalize(query)
        self.assertIn("người lao động", norm)
        self.assertIn("hợp đồng lao động", norm)
        self.assertIn("NLĐ (người lao động)", norm)
        self.assertIn("HĐLĐ (hợp đồng lao động)", norm)
        self.assertEqual(acrs["NLĐ"], "người lao động")
        self.assertEqual(acrs["HĐLĐ"], "hợp đồng lao động")
        self.assertEqual(len(rules), 2)

    def test_replace_mode(self) -> None:
        query = "Thủ tục nộp thuế TNCN tại cơ quan thuế"
        norm, acrs, _ = self.normalizer_replace.normalize(query)
        self.assertEqual(norm, "Thủ tục nộp thuế thuế thu nhập cá nhân tại cơ quan thuế")
        self.assertEqual(acrs["TNCN"], "thuế thu nhập cá nhân")

    def test_dual_space_mode(self) -> None:
        query = "Thời gian tham gia BHXH bắt buộc"
        norm, acrs, _ = self.normalizer_dual.normalize(query)
        self.assertEqual(norm, "Thời gian tham gia BHXH bảo hiểm xã hội bắt buộc")

    def test_case_insensitivity(self) -> None:
        query = "quyền lợi của nld khi tham gia bhyt"
        norm, acrs, _ = self.normalizer_expand.normalize(query)
        self.assertIn("người lao động", norm)
        self.assertIn("bảo hiểm y tế", norm)

    def test_nested_acronym_precedence(self) -> None:
        query = "Thủ tục cấp đổi GCNQSDĐ cho hộ gia đình"
        norm, acrs, _ = self.normalizer_expand.normalize(query)
        self.assertIn("giấy chứng nhận quyền sử dụng đất", norm)
        self.assertIn("GCNQSDĐ", acrs)
        self.assertNotIn("QSDĐ", acrs)

    def test_no_false_positives_in_vietnamese_words(self) -> None:
        query = "Hội đồng nhân dân tỉnh Đồng Nai họp về phát triển nông nghiệp"
        norm, acrs, _ = self.normalizer_expand.normalize(query)
        self.assertNotIn("HĐ", acrs)
        self.assertEqual(norm, query)

    def test_empty_or_whitespace_query(self) -> None:
        norm, acrs, rules = self.normalizer_expand.normalize("   ")
        self.assertEqual(norm, "   ")
        self.assertEqual(acrs, {})
        self.assertEqual(rules, [])


class TestLLMQueryRewriter(unittest.TestCase):
    def test_prompt_construction(self) -> None:
        rewriter = LLMQueryRewriter()
        prompt = rewriter.build_prompt("Doanh nghiệp không đóng BHXH bị phạt gì?")
        self.assertIn("--- VÍ DỤ MINH HỌA ---", prompt)
        self.assertIn("Doanh nghiệp không đóng BHXH bị phạt gì?", prompt)
        self.assertIn("expanded_query", prompt)

    def test_parse_valid_json(self) -> None:
        raw_output = """```json
{
  "intent": "Xử phạt doanh nghiệp trốn đóng bảo hiểm xã hội",
  "statutory_terms": ["trốn đóng bảo hiểm xã hội", "xử phạt vi phạm hành chính", "truy cứu trách nhiệm hình sự"],
  "expanded_query": "Doanh nghiệp không đóng BHXH bị phạt gì trốn đóng bảo hiểm xã hội xử phạt vi phạm hành chính",
  "confidence": "HIGH"
}
```"""
        rewriter = LLMQueryRewriter()
        res = rewriter.parse_and_validate("Doanh nghiệp không đóng BHXH bị phạt gì?", raw_output)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.confidence, "HIGH")
        self.assertEqual(len(res.statutory_terms), 3)
        self.assertIn("trốn đóng bảo hiểm xã hội", res.expanded_query)

    def test_parse_malformed_json(self) -> None:
        raw_output = "Đây là kết quả: {intent: incomplete json..."
        rewriter = LLMQueryRewriter()
        res = rewriter.parse_and_validate("Truy vấn gốc", raw_output)
        self.assertFalse(res.is_valid)
        self.assertIn("JSON parsing error", res.error_message)
        self.assertEqual(res.expanded_query, "Truy vấn gốc")

    def test_information_loss_rejection(self) -> None:
        raw_output = json.dumps({
            "intent": "Luật thủy sản",
            "statutory_terms": ["đánh bắt xa bờ"],
            "expanded_query": "đánh bắt xa bờ tàu cá khai thác hải sản",
            "confidence": "LOW",
        })
        rewriter = LLMQueryRewriter()
        res = rewriter.parse_and_validate("Thủ tục thành lập công ty chứng khoán", raw_output)
        self.assertFalse(res.is_valid)
        self.assertIn("Information loss", res.error_message)
        self.assertEqual(res.expanded_query, "Thủ tục thành lập công ty chứng khoán")


class TestQueryProcessorPipeline(unittest.TestCase):
    def test_pipeline_normalizer_only(self) -> None:
        normalizer = LegalAcronymNormalizer()
        processor = QueryProcessor(normalizer=normalizer, enable_llm=False)

        res = processor.process("NLĐ có quyền đơn phương chấm dứt HĐLĐ không?")
        self.assertEqual(res.status, ProcessingStatus.SUCCESS)
        self.assertIn("người lao động", res.final_query)
        self.assertIn("hợp đồng lao động", res.final_query)
        self.assertGreater(res.metadata.normalizer_latency_ms, 0.0)
        self.assertEqual(res.metadata.llm_latency_ms, 0.0)

    def test_pipeline_with_mock_llm_success(self) -> None:
        def mock_inference(prompt: str) -> str:
            return json.dumps({
                "intent": "Đơn phương chấm dứt hợp đồng lao động của người lao động",
                "statutory_terms": ["quyền đơn phương chấm dứt hợp đồng lao động", "thời hạn báo trước"],
                "expanded_query": "NLĐ (người lao động) có quyền đơn phương chấm dứt HĐLĐ (hợp đồng lao động) không quyền đơn phương chấm dứt thời hạn báo trước",
                "confidence": "HIGH",
            })

        normalizer = LegalAcronymNormalizer()
        rewriter = LLMQueryRewriter(inference_fn=mock_inference)
        processor = QueryProcessor(normalizer=normalizer, rewriter=rewriter, enable_llm=True)

        res = processor.process("NLĐ có quyền đơn phương chấm dứt HĐLĐ không?")
        self.assertEqual(res.status, ProcessingStatus.SUCCESS)
        self.assertIsNone(res.failure_type)
        self.assertIn("thời hạn báo trước", res.final_query)
        self.assertEqual(len(res.statutory_terms), 2)
        self.assertGreater(res.metadata.total_latency_ms, 0.0)

    def test_pipeline_fallback_on_corrupt_llm(self) -> None:
        def corrupt_inference(prompt: str) -> str:
            return "Xin lỗi, tôi không thể trả lời bằng JSON."

        normalizer = LegalAcronymNormalizer()
        rewriter = LLMQueryRewriter(inference_fn=corrupt_inference)
        processor = QueryProcessor(normalizer=normalizer, rewriter=rewriter, enable_llm=True)

        res = processor.process("NLĐ có quyền đơn phương chấm dứt HĐLĐ không?")
        self.assertEqual(res.status, ProcessingStatus.FALLBACK_NORMALIZED)
        self.assertEqual(res.failure_type, FailureType.MALFORMED_OUTPUT)
        self.assertEqual(res.final_query, res.normalized_query)
        self.assertIn("người lao động", res.final_query)

    def test_pipeline_fallback_raw_when_no_acronyms(self) -> None:
        def corrupt_inference(prompt: str) -> str:
            return "Corrupted output."

        normalizer = LegalAcronymNormalizer()
        rewriter = LLMQueryRewriter(inference_fn=corrupt_inference)
        processor = QueryProcessor(normalizer=normalizer, rewriter=rewriter, enable_llm=True)

        query = "Thủ tục chuyển nhượng quyền sử dụng đất nông nghiệp"
        res = processor.process(query)
        self.assertEqual(res.status, ProcessingStatus.FALLBACK_RAW)
        self.assertEqual(res.failure_type, FailureType.MALFORMED_OUTPUT)
        self.assertEqual(res.final_query, query)


if __name__ == "__main__":
    unittest.main()
