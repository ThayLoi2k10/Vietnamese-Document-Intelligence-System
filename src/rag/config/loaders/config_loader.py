"""Configuration loader for JSON, YAML, and dictionary inputs."""

import json
from pathlib import Path
from typing import Any, Dict, Union

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
from src.rag.config.validation.validator import ConfigValidator
from src.rag.errors.exceptions import ConfigurationValidationError


class ConfigLoader:
    """Loads and instantiates validated ExperimentConfig instances."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any], validate: bool = True) -> ExperimentConfig:
        try:
            exp_id = str(data["experiment_id"])

            ds_data = data.get("dataset", {})
            dataset_cfg = DatasetConfig(
                split_role=SplitRole(ds_data.get("split_role", "DEVELOPMENT")),
                dataset_name=ds_data.get("dataset_name", "retrieval_benchmark"),
                split_path=ds_data.get("split_path", "data/benchmarks/retrieval_train.json"),
                sample_limit=ds_data.get("sample_limit"),
            )

            ret_data = data.get("retriever", {})
            retriever_cfg = RetrieverConfig(
                retriever_name=ret_data.get("retriever_name", "phase8_vector_store"),
                retrieval_top_k=int(ret_data.get("retrieval_top_k", 10)),
                retrieval_mode=ret_data.get("retrieval_mode", "CHUNK"),
                component_version=ret_data.get("component_version", "retriever-v1.0"),
                parameters=ret_data.get("parameters", {}),
            )

            rerank_data = data.get("reranker", {})
            reranker_cfg = RerankerConfig(
                enabled=bool(rerank_data.get("enabled", False)),
                model_name=rerank_data.get("model_name"),
                reranker_top_k=int(rerank_data.get("reranker_top_k", 10)),
                component_version=rerank_data.get("component_version", "reranker-v1.0"),
                parameters=rerank_data.get("parameters", {}),
            )

            ctx_data = data.get("context", {})
            context_cfg = ContextConfig(
                context_top_k=int(ctx_data.get("context_top_k", 5)),
                score_threshold=ctx_data.get("score_threshold"),
                max_context_items=ctx_data.get("max_context_items"),
                deduplication_policy=ctx_data.get("deduplication_policy", "EXACT_CHUNK_ID"),
                deduplication_options=ctx_data.get("deduplication_options", {}),
                expansion_policy=ctx_data.get("expansion_policy", "DISABLED"),
                expansion_options=ctx_data.get("expansion_options", {}),
                ordering_strategy=ctx_data.get("ordering_strategy", "SCORE_DESCENDING"),
                max_context_tokens=int(ctx_data.get("max_context_tokens", 4096)),
                budget_overflow_policy=ctx_data.get("budget_overflow_policy", "DROP_LOWEST_PRIORITY_ITEMS"),
                component_version=ctx_data.get("component_version", "context-v1.0"),
            )

            p_data = data.get("prompting", {})
            prompt_cfg = PromptConfig(
                prompt_version=p_data.get("prompt_version", "prompt_v1"),
                template_name=p_data.get("template_name", "standard_legal_qa_v1"),
                instruction_policy_version=p_data.get("instruction_policy_version", "instructions-v1.0"),
            )

            gen_data = data.get("generation", {})
            gen_cfg = GenerationConfig(
                llm_model=gen_data.get("llm_model", "Qwen/Qwen2.5-7B-Instruct"),
                thinking_mode=bool(gen_data.get("thinking_mode", False)),
                max_output_tokens=int(gen_data.get("max_output_tokens", 1024)),
                temperature=float(gen_data.get("temperature", 0.0)),
                top_p=float(gen_data.get("top_p", 1.0)),
                seed=gen_data.get("seed", 42),
                component_version=gen_data.get("component_version", "generator-v1.0"),
                extra_parameters=gen_data.get("extra_parameters", {}),
            )

            eval_data = data.get("evaluation", {})
            eval_cfg = EvaluationConfig(
                evaluator_names=eval_data.get(
                    "evaluator_names",
                    ["citation_precision", "citation_recall", "abstention_correctness", "faithfulness"],
                ),
                evaluation_cutoff=int(eval_data.get("evaluation_cutoff", 10)),
                component_version=eval_data.get("component_version", "eval-v1.0"),
            )

            rep_data = data.get("reproducibility", {})
            rep_cfg = ReproducibilityConfig(
                seed=int(rep_data.get("seed", 42)),
                code_version=rep_data.get("code_version", "git_commit_placeholder"),
                notes=rep_data.get("notes"),
            )

            cfg = ExperimentConfig(
                experiment_id=exp_id,
                dataset=dataset_cfg,
                retriever=retriever_cfg,
                reranker=reranker_cfg,
                context=context_cfg,
                prompting=prompt_cfg,
                generation=gen_cfg,
                evaluation=eval_cfg,
                reproducibility=rep_cfg,
                is_tuning_experiment=bool(data.get("is_tuning_experiment", False)),
                base_config_id=data.get("base_config_id"),
                changed_parameters=data.get("changed_parameters", {}),
            )
        except KeyError as err:
            raise ConfigurationValidationError(f"Missing required configuration field: {err}")

        if validate:
            ConfigValidator.validate(cfg)

        return cfg

    @classmethod
    def from_file(cls, file_path: Union[str, Path], validate: bool = True) -> ExperimentConfig:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        content = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            data = json.loads(content)
        elif path.suffix.lower() in [".yaml", ".yml"]:
            try:
                import yaml
                data = yaml.safe_load(content)
            except ImportError:
                raise ImportError("PyYAML is not installed to read .yaml files.")
        else:
            data = json.loads(content)

        return cls.from_dict(data, validate=validate)
