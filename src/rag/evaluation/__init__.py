"""Evaluation metrics, backends, and report generators for RAG systems."""

from src.rag.evaluation.backends import (
    HeuristicFaithfulnessEvaluator,
    RuleBasedAbstentionCorrectnessEvaluator,
    RuleBasedCitationPrecisionEvaluator,
    RuleBasedCitationRecallEvaluator,
)
from src.rag.evaluation.interfaces import (
    AbstentionCorrectnessEvaluator,
    AnswerRelevanceEvaluator,
    BaseGenerationEvaluator,
    CitationPrecisionEvaluator,
    CitationRecallEvaluator,
    FaithfulnessEvaluator,
    RAGEvaluator,
    UnsupportedClaimEvaluator,
)
from src.rag.evaluation.metrics import (
    EvaluationInput,
    EvaluationReport,
)

__all__ = [
    "EvaluationInput",
    "EvaluationReport",
    "BaseGenerationEvaluator",
    "FaithfulnessEvaluator",
    "AnswerRelevanceEvaluator",
    "CitationPrecisionEvaluator",
    "CitationRecallEvaluator",
    "AbstentionCorrectnessEvaluator",
    "UnsupportedClaimEvaluator",
    "RAGEvaluator",
    "RuleBasedCitationPrecisionEvaluator",
    "RuleBasedCitationRecallEvaluator",
    "RuleBasedAbstentionCorrectnessEvaluator",
    "HeuristicFaithfulnessEvaluator",
]
