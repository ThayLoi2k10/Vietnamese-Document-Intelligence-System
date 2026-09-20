# tests/unit/evaluation/test_citation_evaluators.py
"""Unit tests cho các bộ đánh giá Citation (Precision và Recall) trong Phase 9 RAG.

Tuân thủ Mục 17: Evaluation Architecture và Mục 22: Testing Strategy.
"""

import unittest

from src.rag.abstention.detection.models import (
    AbstentionDecision,
    AbstentionReasonCode,
)
from src.rag.citation.validation.models import CitationValidationResult
from src.rag.context.domain.models import AssembledContext, ContextItem
from src.rag.evaluation.backends.rule_based_evaluator import (
    RuleBasedCitationPrecisionEvaluator,
    RuleBasedCitationRecallEvaluator,
)
from src.rag.evaluation.metrics.models import EvaluationInput
from src.rag.generation.domain.models import FinishStatus, GenerationResult
from src.rag.output.schemas.response_schema import (
    Citation,
    ConfidenceLevel,
    LegalAnswer,
)
from src.rag.pipeline.models import RAGResponse


class TestCitationEvaluators(unittest.TestCase):
    """Kiểm tra tính chính xác của CitationPrecisionEvaluator và CitationRecallEvaluator."""

    def test_citation_evaluation_resolves_via_assembled_context(self) -> None:
        """Chứng minh evaluator không cần LLM xuất global_article_id mà vẫn đánh giá đúng thông qua assembled_context provenance."""
        # 1. Assembled Context
        ctx_item1 = ContextItem(
            context_item_id="[1]",
            source_identity="chk_1",
            law_id="59/2020/QH14",
            text="Nội dung điều luật 1",
            source_rank=1,
            context_position=1,
            global_article_id=55472,
        )
        ctx_item2 = ContextItem(
            context_item_id="[2]",
            source_identity="chk_2",
            law_id="68/2014/QH13",
            text="Nội dung điều luật 2",
            source_rank=2,
            context_position=2,
            global_article_id=26197,
        )
        assembled_context = AssembledContext(
            query="Test query",
            items=[ctx_item1, ctx_item2],
            total_tokens=100,
        )

        # 2. LLM Citations (chỉ chứa citation_number và text law_id do LLM sinh ra, không có internal global_article_id)
        citations = [
            Citation(
                citation_number=1,
                law_id="Văn bản 59/2020/QH14, 55472",
            ),
            Citation(
                citation_number=2,
                law_id="Văn bản 68/2014/QH13, 26197",
            ),
        ]
        legal_answer = LegalAnswer(
            answer="Câu trả lời pháp lý từ LLM.",
            citations=citations,
            confidence_level=ConfidenceLevel.FULL,
        )

        rag_response = RAGResponse(
            question="Test query",
            legal_answer=legal_answer,
            citation_validation=CitationValidationResult(
                is_valid=True,
                valid_citations=citations,
            ),
            abstention_decision=AbstentionDecision(
                status=ConfidenceLevel.FULL,
                reason_code=AbstentionReasonCode.SUFFICIENT_EVIDENCE,
            ),
            assembled_context=assembled_context,
            generation_result=GenerationResult(
                model_identifier="test-model",
                raw_response="",
                generation_metadata={},
                usage_metadata={},
                finish_status=FinishStatus.COMPLETED,
            ),
        )

        eval_input = EvaluationInput(
            query="Test query",
            rag_response=rag_response,
            ground_truth_article_ids=[55472],
            ground_truth_law_ids=[],
        )

        precision_evaluator = RuleBasedCitationPrecisionEvaluator()
        recall_evaluator = RuleBasedCitationRecallEvaluator()

        precision = precision_evaluator.evaluate_sample(eval_input)
        recall = recall_evaluator.evaluate_sample(eval_input)

        self.assertEqual(precision, 0.5)
        self.assertEqual(recall, 1.0)


if __name__ == "__main__":
    unittest.main()
