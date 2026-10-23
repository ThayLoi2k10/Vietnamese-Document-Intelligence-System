"""Integration tests for LLMOnlyPipeline zero-context execution and response contracts."""

import unittest

from src.rag.abstention.detection.abstention_detector import AbstentionDetector
from src.rag.abstention.policies.abstention_validator import AbstentionValidator
from src.rag.citation.validation.citation_validator import CitationValidator
from src.rag.errors.exceptions import (
    InvalidPromptError,
    MalformedStructuredOutputError,
)
from src.rag.generation.mocks.mock_generator import MockLLMGenerator
from src.rag.output.parsers.structured_parser import OutputParser
from src.rag.output.schemas.response_schema import ConfidenceLevel
from src.rag.pipeline.llm_only_pipeline import LLMOnlyPipeline
from src.rag.pipeline.models import RAGResponse
from src.rag.prompting.builders.prompt_builder import PromptBuilder


class TestLLMOnlyPipelineIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.prompt_builder = PromptBuilder()
        self.output_parser = OutputParser()
        self.citation_validator = CitationValidator()
        self.abstention_detector = AbstentionDetector()
        self.abstention_validator = AbstentionValidator()

    def test_llm_only_happy_path(self) -> None:
        custom_response = {
            "answer": "Theo Bộ luật Dân sự, hợp đồng là sự thỏa thuận giữa các bên.",
            "citations": [],
            "confidence_level": "FULL",
            "abstention_reason": None,
        }
        generator = MockLLMGenerator(custom_json_data=custom_response)

        pipeline = LLMOnlyPipeline(
            prompt_builder=self.prompt_builder,
            generator=generator,
            output_parser=self.output_parser,
            abstention_detector=self.abstention_detector,
            abstention_validator=self.abstention_validator,
            citation_validator=self.citation_validator,
        )

        response: RAGResponse = pipeline.run(
            query="Thế nào là hợp đồng theo pháp luật Việt Nam?",
        )

        self.assertEqual(
            response.question,
            "Thế nào là hợp đồng theo pháp luật Việt Nam?",
        )
        self.assertEqual(
            response.legal_answer.answer,
            "Theo Bộ luật Dân sự, hợp đồng là sự thỏa thuận giữa các bên.",
        )
        self.assertEqual(
            response.legal_answer.confidence_level, ConfidenceLevel.FULL
        )
        self.assertEqual(response.legal_answer.citations, [])
        self.assertTrue(response.citation_validation.is_valid)
        self.assertEqual(
            response.abstention_decision.status, ConfidenceLevel.FULL
        )

        self.assertEqual(len(response.assembled_context.items), 0)
        self.assertEqual(response.execution_metadata["retrieval_items_count"], 0)
        self.assertEqual(response.execution_metadata["context_items_count"], 0)
        self.assertEqual(response.execution_metadata["system_type"], "LLM_ONLY")

        self.assertIsNotNone(generator.last_request)
        sent_user_prompt = generator.last_request.prompt.user_prompt
        self.assertNotIn("CĂN CỨ PHÁP LÝ ĐƯỢC CUNG CẤP", sent_user_prompt)
        self.assertIn("Thế nào là hợp đồng theo pháp luật Việt Nam?", sent_user_prompt)

        public_dict = response.to_public_dict()
        self.assertIn("answer", public_dict)
        self.assertIn("citations", public_dict)
        self.assertNotIn("assembled_context", public_dict)
        self.assertNotIn("retrieval_items_count", public_dict)

    def test_llm_only_abstention(self) -> None:
        custom_response = {
            "answer": "Tôi không có đủ thông tin chi tiết về thông tư này.",
            "citations": [],
            "confidence_level": "INSUFFICIENT",
            "abstention_reason": "Thiếu dữ liệu văn bản hướng dẫn cụ thể",
        }
        generator = MockLLMGenerator(custom_json_data=custom_response)

        pipeline = LLMOnlyPipeline(
            prompt_builder=self.prompt_builder,
            generator=generator,
            output_parser=self.output_parser,
            abstention_detector=self.abstention_detector,
            abstention_validator=self.abstention_validator,
        )

        response: RAGResponse = pipeline.run(
            query="Thông tư số 999/2026/TT-BTP quy định điều gì?",
        )

        self.assertEqual(
            response.legal_answer.confidence_level,
            ConfidenceLevel.INSUFFICIENT,
        )
        self.assertEqual(
            response.abstention_decision.status,
            ConfidenceLevel.INSUFFICIENT,
        )
        self.assertEqual(
            response.legal_answer.abstention_reason,
            "Thiếu dữ liệu văn bản hướng dẫn cụ thể",
        )

    def test_llm_only_malformed_output(self) -> None:
        generator = MockLLMGenerator(mode=MockLLMGenerator.MODE_MALFORMED_JSON)

        pipeline = LLMOnlyPipeline(
            prompt_builder=self.prompt_builder,
            generator=generator,
            output_parser=self.output_parser,
            abstention_detector=self.abstention_detector,
        )

        with self.assertRaises(MalformedStructuredOutputError):
            pipeline.run(query="Câu hỏi kiểm tra lỗi syntax JSON.")

    def test_llm_only_empty_query(self) -> None:
        generator = MockLLMGenerator()
        pipeline = LLMOnlyPipeline(
            prompt_builder=self.prompt_builder,
            generator=generator,
            output_parser=self.output_parser,
            abstention_detector=self.abstention_detector,
        )

        with self.assertRaises(InvalidPromptError):
            pipeline.run(query="   ")


if __name__ == "__main__":
    unittest.main()
