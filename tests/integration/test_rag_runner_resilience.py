# tests/integration/test_rag_runner_resilience.py
"""Integration tests cho cơ chế cô lập lỗi (Error Isolation) và Checkpoint/Resume trong Phase 9 RAG Runner.

Tuân thủ:
    - Mục 16: RAG Pipeline
    - Mục 20: Error Handling (Typed, Explicit, Non-silent, Stage-attributable)
    - Mục 22: Testing Strategy (Mock/Fake strategy, không gọi model thật hay GPU)
"""

import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional
import unittest

from scripts.evaluation.run_rag_experiment import (
    load_checkpoint,
    reconstruct_rag_response_from_dict,
    save_checkpoint,
)
from src.rag.abstention.detection.abstention_detector import AbstentionDetector
from src.rag.abstention.policies.abstention_validator import AbstentionValidator
from src.rag.citation.validation.citation_validator import CitationValidator
from src.rag.context.context_assembler import ContextAssembler
from src.rag.errors.exceptions import (
    LLMGenerationError,
    MalformedStructuredOutputError,
)
from src.rag.evaluation.backends.rule_based_evaluator import (
    HeuristicFaithfulnessEvaluator,
    RuleBasedAbstentionCorrectnessEvaluator,
    RuleBasedCitationPrecisionEvaluator,
    RuleBasedCitationRecallEvaluator,
)
from src.rag.evaluation.interfaces.evaluator import RAGEvaluator
from src.rag.evaluation.metrics.models import EvaluationInput
from src.rag.generation.domain.models import (
    FinishStatus,
    GenerationRequest,
    GenerationResult,
)
from src.rag.generation.interfaces.generator import LLMGenerator
from src.rag.output.parsers.structured_parser import OutputParser
from src.rag.pipeline.models import RAGResponse
from src.rag.pipeline.rag_pipeline import RAGPipeline
from src.rag.prompting.builders.prompt_builder import PromptBuilder
from src.rag.retrieval.adapters.mock_retriever import MockRetriever
from tests.fixtures.sample_retrieval import make_sample_retrieval_items


class SequenceMockLLMGenerator(LLMGenerator):
    """Mock Generator trả về kết quả theo danh sách kịch bản tuần tự cho từng lượt gọi."""

    def __init__(self, responses: List[Any]) -> None:
        self.responses = responses
        self.call_count = 0

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if self.call_count >= len(self.responses):
            raw = "{}"
        else:
            item = self.responses[self.call_count]
            if isinstance(item, Exception):
                self.call_count += 1
                raise item
            raw = item
        self.call_count += 1
        return GenerationResult(
            model_identifier=request.model_identifier,
            raw_response=raw,
            generation_metadata={},
            usage_metadata={},
            finish_status=FinishStatus.COMPLETED,
        )


class TestRAGRunnerResilience(unittest.TestCase):
    """Kiểm thử tính bền vững và khả năng phục hồi lỗi của quy trình thực nghiệm RAG."""

    def setUp(self) -> None:
        self.sample_items = make_sample_retrieval_items()
        self.retriever = MockRetriever(canned_items=self.sample_items)
        self.context_assembler = ContextAssembler()
        self.prompt_builder = PromptBuilder()
        self.output_parser = OutputParser()
        self.citation_validator = CitationValidator()
        self.abstention_detector = AbstentionDetector()
        self.abstention_validator = AbstentionValidator()

    def _build_pipeline(self, generator: LLMGenerator) -> RAGPipeline:
        return RAGPipeline(
            retriever=self.retriever,
            reranker=None,
            context_assembler=self.context_assembler,
            prompt_builder=self.prompt_builder,
            generator=generator,
            output_parser=self.output_parser,
            citation_validator=self.citation_validator,
            abstention_detector=self.abstention_detector,
            abstention_validator=self.abstention_validator,
        )

    def test_expected_structured_output_failure_isolation_and_continue(self) -> None:
        """Kiểm tra: Khi một câu hỏi gặp lỗi cấu trúc đầu ra (MalformedStructuredOutputError),
        hệ thống ghi nhận FAILED rõ ràng, lưu checkpoint và tiếp tục xử lý các câu hỏi tiếp theo
        thay vì làm gián đoạn toàn bộ thực nghiệm.
        """
        valid_json = json.dumps({
            "answer": "Câu trả lời hợp lệ.",
            "citations": [
                {
                    "citation_number": 1,
                    "law_id": "59/2020/QH14",
                    "article_identifier": "Điều 1",
                    "supporting_evidence_or_quote": "nội dung căn cứ",
                }
            ],
            "confidence_level": "FULL",
            "abstention_reason": None,
        })
        malformed_output = "Đây là văn bản tự do không phải JSON hợp lệ: { 'answer': ..."

        # Kịch bản 3 câu hỏi: Q1 thành công -> Q2 lỗi cú pháp -> Q3 thành công
        generator = SequenceMockLLMGenerator(
            responses=[valid_json, malformed_output, valid_json]
        )
        pipeline = self._build_pipeline(generator)

        queries = [
            {"query_id": 1, "query_text": "Câu hỏi số 1", "relevant_article_ids": [55472]},
            {"query_id": 2, "query_text": "Câu hỏi số 2", "relevant_article_ids": [55472]},
            {"query_id": 3, "query_text": "Câu hỏi số 3", "relevant_article_ids": [55472]},
        ]

        predictions: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        evaluation_inputs: List[EvaluationInput] = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_file = Path(tmp_dir) / "test_checkpoint.json"

            for q_item in queries:
                q_id = q_item["query_id"]
                q_text = q_item["query_text"]
                try:
                    resp = pipeline.run(query=q_text)
                    predictions.append({
                        "query_id": q_id,
                        "query": q_text,
                        "status": "COMPLETED",
                        "response": resp.to_full_audit_dict(),
                    })
                    evaluation_inputs.append(
                        EvaluationInput(
                            query=q_text,
                            rag_response=resp,
                            ground_truth_article_ids=q_item["relevant_article_ids"],
                        )
                    )
                except (MalformedStructuredOutputError, LLMGenerationError) as e:
                    raw_out = getattr(e, "details", {}).get("raw_text")
                    failures.append({
                        "query_id": q_id,
                        "query": q_text,
                        "status": "FAILED",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "raw_output": raw_out,
                    })
                    save_checkpoint(ckpt_file, "test_exp", predictions, failures)

            # Xác minh:
            # 1. Toàn bộ 3 câu hỏi đều được duyệt qua
            self.assertEqual(generator.call_count, 3)
            # 2. Hai câu hỏi thành công (Q1, Q3) nằm trong predictions
            self.assertEqual(len(predictions), 2)
            self.assertEqual([p["query_id"] for p in predictions], [1, 3])
            # 3. Một câu hỏi thất bại (Q2) được ghi nhận tường minh trong failures
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["query_id"], 2)
            self.assertEqual(failures[0]["status"], "FAILED")
            self.assertEqual(failures[0]["error_type"], "MalformedStructuredOutputError")
            self.assertEqual(failures[0]["raw_output"], malformed_output)
            # 4. Checkpoint phản ánh đúng trạng thái
            loaded_preds, loaded_fails = load_checkpoint(ckpt_file)
            self.assertEqual(len(loaded_preds), 1)  # Lưu khi Q2 lỗi, lúc đó chỉ có Q1
            self.assertEqual(len(loaded_fails), 1)
            self.assertEqual(loaded_fails[0]["query_id"], 2)

    def test_unexpected_exception_still_propagates(self) -> None:
        """Kiểm tra: Các ngoại lệ bất thường ngoài dự kiến (như RuntimeError, OOM, lỗi lập trình)
        phải tiếp tục ném ra ngoài và không bị nuốt chửng.
        """
        # Generator ném lỗi hệ thống không mong đợi
        generator = SequenceMockLLMGenerator(
            responses=[RuntimeError("CUDA out of memory (mô phỏng)")]
        )
        pipeline = self._build_pipeline(generator)

        with self.assertRaises(RuntimeError) as ctx:
            pipeline.run(query="Câu hỏi gây lỗi phần cứng")
        self.assertIn("CUDA out of memory", str(ctx.exception))

    def test_checkpoint_persistence_and_resume(self) -> None:
        """Kiểm tra: Khả năng ghi checkpoint và tái nạp (resume) để bỏ qua các câu hỏi đã hoàn thành."""
        valid_json = json.dumps({
            "answer": "Câu trả lời hợp lệ.",
            "citations": [],
            "confidence_level": "FULL",
            "abstention_reason": None,
        })
        generator = SequenceMockLLMGenerator(responses=[valid_json, valid_json])
        pipeline = self._build_pipeline(generator)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_file = Path(tmp_dir) / "checkpoint.json"

            # 1. Chạy câu hỏi 1 và lưu checkpoint
            resp1 = pipeline.run(query="Câu hỏi 1")
            predictions = [{
                "query_id": 1,
                "query": "Câu hỏi 1",
                "status": "COMPLETED",
                "response": resp1.to_full_audit_dict(),
            }]
            failures = [{
                "query_id": 99,
                "query": "Câu hỏi lỗi trước đó",
                "status": "FAILED",
                "error_type": "MalformedStructuredOutputError",
                "error_message": "JSONDecodeError",
                "raw_output": "invalid",
            }]
            save_checkpoint(ckpt_file, "exp_test", predictions, failures)

            # 2. Resume từ checkpoint
            loaded_preds, loaded_fails = load_checkpoint(ckpt_file)
            completed_ids = {p["query_id"] for p in loaded_preds}
            failed_ids = {f["query_id"] for f in loaded_fails}

            self.assertIn(1, completed_ids)
            self.assertIn(99, failed_ids)

            # 3. Chạy tiếp danh sách câu hỏi [1, 2]: câu hỏi 1 phải được bỏ qua
            queries = [
                {"query_id": 1, "query_text": "Câu hỏi 1"},
                {"query_id": 2, "query_text": "Câu hỏi 2"},
            ]
            run_count = 0
            for q in queries:
                qid = q["query_id"]
                if qid in completed_ids:
                    continue  # Bỏ qua câu hỏi đã hoàn thành
                if qid in failed_ids:
                    continue
                pipeline.run(query=q["query_text"])
                run_count += 1

            # Câu hỏi 1 bị bỏ qua, chỉ có câu hỏi 2 được gọi
            self.assertEqual(run_count, 1)
            # Tổng số lần generator được gọi trong test là 2 (1 lần ở bước 1, 1 lần ở bước 3)
            self.assertEqual(generator.call_count, 2)

    def test_reconstruct_rag_response_from_dict(self) -> None:
        """Kiểm tra: Tái tạo RAGResponse từ dict checkpoint đảm bảo đầy đủ trường cho evaluator."""
        valid_json = json.dumps({
            "answer": "Người điều khiển xe phải chấp hành hiệu lệnh.",
            "citations": [
                {
                    "citation_number": 1,
                    "law_id": "59/2020/QH14",
                    "article_identifier": "Điều 5",
                    "supporting_evidence_or_quote": "chấp hành hiệu lệnh",
                }
            ],
            "confidence_level": "FULL",
            "abstention_reason": None,
        })
        generator = SequenceMockLLMGenerator(responses=[valid_json])
        pipeline = self._build_pipeline(generator)

        resp = pipeline.run(query="Hiệu lệnh giao thông?")
        audit_dict = resp.to_full_audit_dict()

        reconstructed_resp = reconstruct_rag_response_from_dict(audit_dict)

        self.assertEqual(reconstructed_resp.question, resp.question)
        self.assertEqual(
            reconstructed_resp.legal_answer.answer, resp.legal_answer.answer
        )
        self.assertEqual(
            len(reconstructed_resp.legal_answer.citations),
            len(resp.legal_answer.citations),
        )
        self.assertEqual(
            reconstructed_resp.legal_answer.confidence_level,
            resp.legal_answer.confidence_level,
        )
        self.assertEqual(
            len(reconstructed_resp.assembled_context.items),
            len(resp.assembled_context.items),
        )

        # Kiểm tra evaluator có thể đánh giá mẫu được tái tạo mà không gặp lỗi
        eval_in = EvaluationInput(
            query="Hiệu lệnh giao thông?",
            rag_response=reconstructed_resp,
            ground_truth_article_ids=[55472],
            ground_truth_law_ids=["59/2020/QH14"],
        )
        evaluator = RAGEvaluator(
            evaluators=[
                RuleBasedCitationPrecisionEvaluator(),
                RuleBasedCitationRecallEvaluator(),
                RuleBasedAbstentionCorrectnessEvaluator(),
                HeuristicFaithfulnessEvaluator(),
            ]
        )
        report = evaluator.evaluate([eval_in], experiment_id="test_rec")
        self.assertEqual(report.sample_count, 1)
        self.assertIn("citation_precision", report.metric_results)
        self.assertIn("faithfulness", report.metric_results)


if __name__ == "__main__":
    unittest.main()
