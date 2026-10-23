"""Domain models for evaluation inputs and experiment reports."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.rag.pipeline.models import RAGResponse


@dataclass
class EvaluationInput:
    query: str
    rag_response: RAGResponse
    ground_truth_answer: Optional[str] = None
    ground_truth_article_ids: List[int] = field(default_factory=list)
    ground_truth_law_ids: List[str] = field(default_factory=list)
    is_answerable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    experiment_id: str
    dataset_split: str
    sample_count: int
    metric_results: Dict[str, float]
    metric_configurations: Dict[str, Any] = field(default_factory=dict)
    component_versions: Dict[str, str] = field(default_factory=dict)
    error_summary: Dict[str, Any] = field(default_factory=dict)
    reproducibility_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
