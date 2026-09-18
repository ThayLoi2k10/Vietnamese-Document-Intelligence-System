# src/rag/evaluation/interfaces/evaluator.py
"""Interface trừu tượng cho hệ thống đánh giá Phase 9 RAG.

Tuân thủ Mục 17: Evaluation Architecture:
    - 17.1 Phân tách hệ thống đánh giá khỏi generation orchestration
    - 17.2 RAGEvaluator interface
    - 17.4 Các evaluator chuyên biệt (Faithfulness, AnswerRelevance, CitationPrecision/Recall, AbstentionCorrectness, UnsupportedClaim)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.rag.evaluation.metrics.models import EvaluationInput, EvaluationReport


class BaseGenerationEvaluator(ABC):
    """Giao diện cơ sở cho một độ đo đánh giá câu trả lời thế hệ RAG."""

    @property
    @abstractmethod
    def metric_name(self) -> str:
        pass

    @abstractmethod
    def evaluate_sample(self, item: EvaluationInput) -> float:
        """Đánh giá một mẫu đơn lẻ và trả về điểm số [0.0, 1.0]."""
        pass

    def evaluate_batch(self, items: List[EvaluationInput]) -> List[float]:
        return [self.evaluate_sample(item) for item in items]


class FaithfulnessEvaluator(BaseGenerationEvaluator):
    """Đánh giá độ trung thực (Faithfulness / Groundedness) của câu trả lời đối với bằng chứng."""

    @property
    def metric_name(self) -> str:
        return "faithfulness"


class AnswerRelevanceEvaluator(BaseGenerationEvaluator):
    """Đánh giá mức độ liên quan và giải quyết câu hỏi của câu trả lời."""

    @property
    def metric_name(self) -> str:
        return "answer_relevance"


class CitationPrecisionEvaluator(BaseGenerationEvaluator):
    """Đánh giá độ chính xác của các trích dẫn (tỷ lệ trích dẫn thực sự liên quan/chính xác)."""

    @property
    def metric_name(self) -> str:
        return "citation_precision"


class CitationRecallEvaluator(BaseGenerationEvaluator):
    """Đánh giá độ bao phủ của các trích dẫn so với ground truth."""

    @property
    def metric_name(self) -> str:
        return "citation_recall"


class AbstentionCorrectnessEvaluator(BaseGenerationEvaluator):
    """Đánh giá tính chính xác của quyết định từ chối trả lời."""

    @property
    def metric_name(self) -> str:
        return "abstention_correctness"


class UnsupportedClaimEvaluator(BaseGenerationEvaluator):
    """Đánh giá tỷ lệ các tuyên bố trong câu trả lời không có căn cứ chứng minh."""

    @property
    def metric_name(self) -> str:
        return "unsupported_claim_rate"


class RAGEvaluator:
    """Bộ điều phối đánh giá tổng hợp của Phase 9."""

    def __init__(
        self,
        evaluators: Optional[List[BaseGenerationEvaluator]] = None,
        evaluator_version: str = "rag-eval-v1.0",
    ) -> None:
        self.evaluators = evaluators or []
        self.evaluator_version = evaluator_version

    def evaluate(
        self,
        dataset: List[EvaluationInput],
        experiment_id: str = "exp_001",
        dataset_split: str = "VALIDATION",
        component_versions: Optional[Dict[str, str]] = None,
    ) -> EvaluationReport:
        """Thực thi đánh giá trên toàn bộ tập dữ liệu mẫu."""
        metric_results: Dict[str, float] = {}
        sample_count = len(dataset)

        for ev in self.evaluators:
            scores = ev.evaluate_batch(dataset)
            mean_score = sum(scores) / len(scores) if scores else 0.0
            metric_results[ev.metric_name] = float(mean_score)

        return EvaluationReport(
            experiment_id=experiment_id,
            dataset_split=dataset_split,
            sample_count=sample_count,
            metric_results=metric_results,
            metric_configurations={
                "evaluator_version": self.evaluator_version,
                "active_metrics": [ev.metric_name for ev in self.evaluators],
            },
            component_versions=component_versions or {},
        )
