# tests/fixtures/sample_configs.py
"""Fixtures cấu hình cho kiểm thử Phase 9 RAG."""

from src.rag.config.schemas.config_models import (
    ContextConfig,
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    GenerationConfig,
    PromptConfig,
    ReproducibilityConfig,
    RerankerConfig,
    RetrieverConfig,
    SplitRole,
)


def make_valid_experiment_config(
    split_role: SplitRole = SplitRole.DEVELOPMENT,
    is_tuning: bool = False,
    reranker_enabled: bool = False,
) -> ExperimentConfig:
    """Tạo đối tượng ExperimentConfig hợp lệ cho kiểm thử."""
    return ExperimentConfig(
        experiment_id="test_exp_001",
        is_tuning_experiment=is_tuning,
        dataset=DatasetConfig(
            split_role=split_role,
            dataset_name="test_dataset",
            split_path="data/benchmarks/retrieval_train.json",
        ),
        retriever=RetrieverConfig(
            retriever_name="mock_retriever",
            retrieval_top_k=20,
            retrieval_mode="CHUNK",
        ),
        reranker=RerankerConfig(
            enabled=reranker_enabled,
            model_name="BAAI/bge-reranker-v2-m3" if reranker_enabled else None,
            reranker_top_k=15 if reranker_enabled else 10,
        ),
        context=ContextConfig(
            context_top_k=5,
            max_context_tokens=4096,
        ),
        prompting=PromptConfig(
            prompt_version="prompt_v1",
        ),
        generation=GenerationConfig(
            llm_model="Qwen/Qwen2.5-7B-Instruct",
            max_output_tokens=1024,
        ),
        evaluation=EvaluationConfig(),
        reproducibility=ReproducibilityConfig(
            seed=42,
            code_version="test-commit",
        ),
    )
