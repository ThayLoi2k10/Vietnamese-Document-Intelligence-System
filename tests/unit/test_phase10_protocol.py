"""Unit tests for Phase 10 LLM-only protocol, zero-context prompts, and configs."""

import json
from pathlib import Path
import unittest

from src.rag.config.loaders.config_loader import ConfigLoader
from src.rag.config.schemas.config_models import SplitRole
from src.rag.context.domain.models import AssembledContext
from src.rag.evaluation.backends.rule_based_evaluator import (
    RuleBasedAbstentionCorrectnessEvaluator,
)
from src.rag.evaluation.metrics.models import EvaluationInput
from src.rag.output.schemas.response_schema import (
    ConfidenceLevel,
    LegalAnswer,
)
from src.rag.pipeline.models import RAGResponse
from src.rag.prompting.builders.prompt_builder import PromptBuilder
from src.rag.prompting.templates.registry import PromptTemplateRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestPhase10Protocol(unittest.TestCase):
    def test_prompt_registry_has_llm_only(self) -> None:
        registry = PromptTemplateRegistry()
        self.assertTrue(registry.has_version("prompt_llm_only_v1"))

        tmpl = registry.get("prompt_llm_only_v1")
        self.assertIn("answer", tmpl.system_template)
        self.assertIn("confidence_level", tmpl.system_template)
        self.assertIn("abstention_reason", tmpl.system_template)
        self.assertNotIn("CĂN CỨ PHÁP LÝ", tmpl.user_template)
        self.assertIn("{query}", tmpl.user_template)

    def test_prompt_builder_zero_context(self) -> None:
        builder = PromptBuilder()
        empty_ctx = AssembledContext(
            query="Thời hạn giải quyết thủ tục là bao lâu?",
            items=[],
            total_tokens=0,
        )
        req = builder.build(
            query="Thời hạn giải quyết thủ tục là bao lâu?",
            context=empty_ctx,
            template_version="prompt_llm_only_v1",
        )
        self.assertEqual(req.prompt_version, "prompt_llm_only_v1")
        self.assertNotIn("CĂN CỨ PHÁP LÝ ĐƯỢC CUNG CẤP", req.user_prompt)
        self.assertIn("Thời hạn giải quyết thủ tục là bao lâu?", req.user_prompt)

    def test_phase10_experiment_config_validity(self) -> None:
        cfg_path = PROJECT_ROOT / "configs" / "experiments" / "phase10_llm_only_experiment.json"
        self.assertTrue(cfg_path.exists())

        config = ConfigLoader.from_file(cfg_path, validate=True)
        self.assertEqual(config.experiment_id, "exp_phase10_llm_only")
        self.assertEqual(config.dataset.split_role, SplitRole.DEVELOPMENT)
        self.assertEqual(config.generation.llm_model, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(config.generation.temperature, 0.0)
        self.assertEqual(config.generation.top_p, 1.0)
        self.assertEqual(config.generation.seed, 42)
        self.assertEqual(config.generation.max_output_tokens, 1024)
        self.assertEqual(config.generation.extra_parameters.get("repetition_penalty"), 1.1)

    def test_abstention_correctness_on_llm_only_response(self) -> None:
        evaluator = RuleBasedAbstentionCorrectnessEvaluator()

        from src.rag.abstention.detection.models import (
            AbstentionDecision,
            AbstentionReasonCode,
        )
        from src.rag.citation.validation.models import CitationValidationResult
        from src.rag.generation.domain.models import FinishStatus, GenerationResult

        ans_full = LegalAnswer(
            answer="Nội dung câu trả lời",
            citations=[],
            confidence_level=ConfidenceLevel.FULL,
            abstention_reason=None,
        )
        resp_full = RAGResponse(
            question="Câu hỏi",
            legal_answer=ans_full,
            citation_validation=CitationValidationResult(is_valid=True),
            abstention_decision=AbstentionDecision(
                status=ConfidenceLevel.FULL,
                reason_code=AbstentionReasonCode.SUFFICIENT_EVIDENCE,
            ),
            assembled_context=AssembledContext(query="Câu hỏi", items=[], total_tokens=0),
            generation_result=GenerationResult(
                model_identifier="Qwen/Qwen2.5-7B-Instruct",
                raw_response="{}",
                finish_status=FinishStatus.COMPLETED,
            ),
        )

        item = EvaluationInput(
            query="Câu hỏi",
            rag_response=resp_full,
            ground_truth_article_ids=[123],
            is_answerable=True,
        )
        score = evaluator.evaluate_sample(item)
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
