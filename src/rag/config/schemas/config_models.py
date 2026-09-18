# src/rag/config/schemas/config_models.py
"""Mô hình cấu hình thực nghiệm hoàn chỉnh (ExperimentConfig) cho Phase 9.

Tuân thủ Mục 18: Experiment Configuration & Mục 19: Dataset / Test Protection:
    - 18.2 Required configuration dimensions
    - 18.3 Configuration object hierarchy
    - 19.5 Configuration-level leakage prevention (SplitRole enum)
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SplitRole(str, Enum):
    """Vai trò của tập dữ liệu trong quy trình nghiên cứu."""

    DEVELOPMENT = "DEVELOPMENT"
    VALIDATION = "VALIDATION"
    FROZEN_TEST = "FROZEN_TEST"


@dataclass
class DatasetConfig:
    """Cấu hình tập dữ liệu truy vấn/đánh giá."""

    split_role: SplitRole
    dataset_name: str = "retrieval_benchmark"
    split_path: str = "data/benchmarks/retrieval_train.json"
    sample_limit: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["split_role"] = self.split_role.value
        return d


@dataclass
class RetrieverConfig:
    """Cấu hình cho thành phần Retriever."""

    retriever_name: str = "phase8_vector_store"
    retrieval_top_k: int = 10
    retrieval_mode: str = "CHUNK"
    component_version: str = "retriever-v1.0"
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RerankerConfig:
    """Cấu hình cho thành phần Reranker."""

    enabled: bool = False
    model_name: Optional[str] = None
    reranker_top_k: int = 10
    component_version: str = "reranker-v1.0"
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContextConfig:
    """Cấu hình cho chính sách tuyển chọn và lắp ráp ngữ cảnh."""

    context_top_k: int = 5
    score_threshold: Optional[float] = None
    max_context_items: Optional[int] = None
    deduplication_policy: str = "EXACT_CHUNK_ID"
    deduplication_options: Dict[str, Any] = field(default_factory=dict)
    expansion_policy: str = "DISABLED"
    expansion_options: Dict[str, Any] = field(default_factory=dict)
    ordering_strategy: str = "SCORE_DESCENDING"
    max_context_tokens: int = 4096
    budget_overflow_policy: str = "DROP_LOWEST_PRIORITY_ITEMS"
    component_version: str = "context-v1.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PromptConfig:
    """Cấu hình cho Prompt Template."""

    prompt_version: str = "prompt_v1"
    template_name: str = "standard_legal_qa_v1"
    instruction_policy_version: str = "instructions-v1.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationConfig:
    """Cấu hình cho mô hình sinh LLM."""

    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    thinking_mode: bool = False
    max_output_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    seed: Optional[int] = 42
    component_version: str = "generator-v1.0"
    extra_parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationConfig:
    """Cấu hình đánh giá."""

    evaluator_names: List[str] = field(
        default_factory=lambda: [
            "citation_precision",
            "citation_recall",
            "abstention_correctness",
            "faithfulness",
        ]
    )
    evaluation_cutoff: int = 10
    component_version: str = "eval-v1.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReproducibilityConfig:
    """Metadata tái lập thực nghiệm."""

    seed: int = 42
    code_version: str = "git_commit_placeholder"
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Cấu hình thực nghiệm cấp cao nhất của Phase 9."""

    experiment_id: str
    dataset: DatasetConfig
    retriever: RetrieverConfig
    reranker: RerankerConfig
    context: ContextConfig
    prompting: PromptConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    reproducibility: ReproducibilityConfig = field(
        default_factory=ReproducibilityConfig
    )
    is_tuning_experiment: bool = False
    base_config_id: Optional[str] = None
    changed_parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "is_tuning_experiment": self.is_tuning_experiment,
            "base_config_id": self.base_config_id,
            "changed_parameters": self.changed_parameters,
            "dataset": self.dataset.to_dict(),
            "retriever": self.retriever.to_dict(),
            "reranker": self.reranker.to_dict(),
            "context": self.context.to_dict(),
            "prompting": self.prompting.to_dict(),
            "generation": self.generation.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "reproducibility": self.reproducibility.to_dict(),
        }
