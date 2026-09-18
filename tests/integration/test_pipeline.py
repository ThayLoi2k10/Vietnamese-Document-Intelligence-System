# tests/integration/test_pipeline.py
"""Integration tests cho RAGPipeline trong Phase 9.

Tuân thủ Mục 22.3: Integration tests:
    - Full pipeline với mock components:
      Mock Retriever -> Mock Reranker -> Real ContextAssembler -> Real PromptBuilder -> Mock LLM -> Real OutputParser -> Real CitationValidator -> Real AbstentionValidator
    - Đảm bảo xác minh toàn diện chu trình RAG không cần tải mô hình thật hay truy vấn DB thật.
"""

import unittest

from src.rag.abstention.detection.abstention_detector import AbstentionDetector
from src.rag.abstention.policies.abstention_validator import AbstentionValidator
from src.rag.citation.validation.citation_validator import CitationValidator
from src.rag.context.context_assembler import ContextAssembler
from src.rag.context.policies.policy_types import ContextPolicy
from src.rag.generation.mocks.mock_generator import MockLLMGenerator
from src.rag.output.parsers.structured_parser import OutputParser
from src.rag.output.schemas.response_schema import ConfidenceLevel
from src.rag.pipeline.models import RAGResponse
from src.rag.pipeline.rag_pipeline import RAGPipeline
from src.rag.prompting.builders.prompt_builder import PromptBuilder
from src.rag.reranking.adapters.mock_reranker import MockReranker
from src.rag.retrieval.adapters.mock_retriever import MockRetriever
from tests.fixtures.sample_retrieval import make_sample_retrieval_items


class TestRAGPipelineIntegration(unittest.TestCase):
    """Kiểm tra tích hợp toàn bộ chu trình xử lý của RAGPipeline."""

    def setUp(self) -> None:
        self.sample_items = make_sample_retrieval_items()
        self.retriever = MockRetriever(canned_items=self.sample_items)
        self.reranker = MockReranker()
        self.context_assembler = ContextAssembler()
        self.prompt_builder = PromptBuilder()
        self.output_parser = OutputParser()
        self.citation_validator = CitationValidator()
        self.abstention_detector = AbstentionDetector()
        self.abstention_validator = AbstentionValidator()

    def test_full_pipeline_happy_path(self) -> None:
        """Kiểm tra chu trình pipeline thành công đầy đủ với MockLLMGenerator trả về answer hợp lệ."""
        generator = MockLLMGenerator(mode=MockLLMGenerator.MODE_VALID_ANSWER)

        pipeline = RAGPipeline(
            retriever=self.retriever,
            reranker=self.reranker,
            context_assembler=self.context_assembler,
            prompt_builder=self.prompt_builder,
            generator=generator,
            output_parser=self.output_parser,
            citation_validator=self.citation_validator,
            abstention_detector=self.abstention_detector,
            abstention_validator=self.abstention_validator,
        )

        response: RAGResponse = pipeline.run(
            query="Người tham gia giao thông chấp hành hiệu lệnh như thế nào?",
            context_policy=ContextPolicy(context_top_k=2),
        )

        # Kiểm tra đối tượng phản hồi RAGResponse
        self.assertEqual(
            response.question,
            "Người tham gia giao thông chấp hành hiệu lệnh như thế nào?",
        )
        self.assertIsNotNone(response.legal_answer.answer)
        self.assertEqual(
            response.legal_answer.confidence_level, ConfidenceLevel.FULL
        )
        self.assertTrue(response.citation_validation.is_valid)
        self.assertEqual(
            response.abstention_decision.status, ConfidenceLevel.FULL
        )
        self.assertEqual(len(response.assembled_context.items), 2)
        self.assertEqual(
            response.execution_metadata["retrieval_items_count"], 3
        )
        self.assertEqual(response.execution_metadata["context_items_count"], 2)

        # Kiểm tra schema xuất bản công khai không chứa vết reasoning
        public_dict = response.to_public_dict()
        self.assertIn("answer", public_dict)
        self.assertIn("citations", public_dict)
        self.assertIn("confidence_level", public_dict)
        self.assertNotIn("internal_metadata", public_dict)
        self.assertNotIn("source_score", public_dict)

    def test_full_pipeline_empty_retrieval_triggers_abstention(self) -> None:
        """Kiểm tra pipeline khi truy xuất không có tài liệu dẫn đến từ chối hợp lệ."""
        empty_retriever = MockRetriever(canned_items=[])
        generator = MockLLMGenerator(mode=MockLLMGenerator.MODE_ABSTENTION)

        pipeline = RAGPipeline(
            retriever=empty_retriever,
            reranker=None,  # Không có reranker
            context_assembler=self.context_assembler,
            prompt_builder=self.prompt_builder,
            generator=generator,
            output_parser=self.output_parser,
            citation_validator=self.citation_validator,
            abstention_detector=self.abstention_detector,
            abstention_validator=self.abstention_validator,
        )

        response: RAGResponse = pipeline.run(
            query="Câu hỏi về vấn đề không có trong luật pháp"
        )

        self.assertEqual(
            response.abstention_decision.status, ConfidenceLevel.INSUFFICIENT
        )
        self.assertEqual(len(response.assembled_context.items), 0)


if __name__ == "__main__":
    unittest.main()
