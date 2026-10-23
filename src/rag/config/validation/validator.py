"""Experiment configuration validator and data leakage gate."""

from src.rag.config.schemas.config_models import ExperimentConfig, SplitRole
from src.rag.errors.exceptions import (
    ConfigurationValidationError,
    EvaluationDataLeakageError,
)


class ConfigValidator:
    """Validates experiment invariants, budget constraints, and dataset leakage."""

    @staticmethod
    def validate(config: ExperimentConfig) -> None:
        if config.dataset.split_role == SplitRole.FROZEN_TEST and config.is_tuning_experiment:
            raise EvaluationDataLeakageError(
                f"VI PHẠM RÒ RỈ DỮ LIỆU: Thực nghiệm '{config.experiment_id}' cấu hình tuning trên FROZEN_TEST.",
                experiment_id=config.experiment_id,
                component_identifier="ConfigValidator",
            )

        ret_k = config.retriever.retrieval_top_k
        ctx_k = config.context.context_top_k

        if ret_k <= 0:
            raise ConfigurationValidationError(f"retrieval_top_k must be > 0, got: {ret_k}")
        if ctx_k <= 0:
            raise ConfigurationValidationError(f"context_top_k must be > 0, got: {ctx_k}")

        if config.reranker.enabled:
            rerank_k = config.reranker.reranker_top_k
            if rerank_k <= 0:
                raise ConfigurationValidationError(f"reranker_top_k must be > 0 when reranker is enabled, got: {rerank_k}")
            if ret_k < rerank_k:
                raise ConfigurationValidationError(
                    f"retrieval_top_k ({ret_k}) cannot be less than reranker_top_k ({rerank_k})"
                )
            if rerank_k < ctx_k:
                raise ConfigurationValidationError(
                    f"reranker_top_k ({rerank_k}) cannot be less than context_top_k ({ctx_k})"
                )
        elif ret_k < ctx_k:
            raise ConfigurationValidationError(
                f"retrieval_top_k ({ret_k}) cannot be less than context_top_k ({ctx_k})"
            )

        if config.context.max_context_tokens <= 0:
            raise ConfigurationValidationError(
                f"max_context_tokens must be > 0, got: {config.context.max_context_tokens}"
            )
        if config.generation.max_output_tokens <= 0:
            raise ConfigurationValidationError(
                f"max_output_tokens must be > 0, got: {config.generation.max_output_tokens}"
            )

        if not config.experiment_id or not config.experiment_id.strip():
            raise ConfigurationValidationError("experiment_id cannot be empty")
        if not config.generation.llm_model or not config.generation.llm_model.strip():
            raise ConfigurationValidationError("llm_model cannot be empty")
        if not config.prompting.prompt_version or not config.prompting.prompt_version.strip():
            raise ConfigurationValidationError("prompt_version cannot be empty")
