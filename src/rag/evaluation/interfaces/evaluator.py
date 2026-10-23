"""Abstract generation evaluators and orchestrators for RAG performance."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from src.rag.evaluation.metrics.models import EvaluationInput, EvaluationReport


class BaseGenerationEvaluator(ABC):
    """Base interface for single-metric generation evaluation."""

    @property
    @abstractmethod
    def metric_name(self) -> str:
        pass

    @abstractmethod
    def evaluate_sample(self, item: EvaluationInput) -> float:
        pass

    def evaluate_batch(self, items: List[EvaluationInput]) -> List[float]:
        return [self.evaluate_sample(item) for item in items]


class FaithfulnessEvaluator(BaseGenerationEvaluator):
    @property
    def metric_name(self) -> str:
        return "faithfulness"


class AnswerRelevanceEvaluator(BaseGenerationEvaluator):
    @property
    def metric_name(self) -> str:
        return "answer_relevance"


class CitationPrecisionEvaluator(BaseGenerationEvaluator):
    @property
    def metric_name(self) -> str:
        return "citation_precision"


class CitationRecallEvaluator(BaseGenerationEvaluator):
    @property
    def metric_name(self) -> str:
        return "citation_recall"


class AbstentionCorrectnessEvaluator(BaseGenerationEvaluator):
    @property
    def metric_name(self) -> str:
        return "abstention_correctness"


class UnsupportedClaimEvaluator(BaseGenerationEvaluator):
    @property
    def metric_name(self) -> str:
        return "unsupported_claim_rate"


class RAGEvaluator:
    """Orchestrator coordinating execution across multiple evaluators."""

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
