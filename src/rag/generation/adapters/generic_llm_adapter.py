"""Generic LLM adapter bridging inference functions to the generator interface."""

from typing import Any, Callable, Dict, Optional

from src.rag.errors.exceptions import (
    LLMGenerationError,
    UnsupportedModelConfigurationError,
)
from src.rag.generation.domain.models import (
    FinishStatus,
    GenerationRequest,
    GenerationResult,
)
from src.rag.generation.interfaces.generator import LLMGenerator


class GenericLLMAdapter(LLMGenerator):
    """General inference adapter compatible with API and local model backends."""

    def __init__(
        self,
        inference_fn: Optional[Callable[[str, str, Dict[str, Any]], str]] = None,
        component_version: str = "generic-llm-v1.0",
        supported_models: Optional[list[str]] = None,
    ) -> None:
        self.inference_fn = inference_fn
        self.component_version = component_version
        self.supported_models = supported_models

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if (
            self.supported_models
            and request.model_identifier not in self.supported_models
        ):
            raise UnsupportedModelConfigurationError(
                f"Model '{request.model_identifier}' is not supported: {self.supported_models}"
            )

        if self.inference_fn is None:
            raise LLMGenerationError(
                "GenericLLMAdapter requires an inference_fn to execute."
            )

        try:
            call_params = {
                "max_tokens": request.max_output_tokens,
                "temperature": request.generation_parameters.temperature,
                "top_p": request.generation_parameters.top_p,
                "seed": request.generation_parameters.seed,
                "stop": request.generation_parameters.stop_sequences,
                "thinking_mode": request.thinking_mode,
                **request.generation_parameters.extra_params,
            }
            raw_text = self.inference_fn(
                request.prompt.system_prompt,
                request.prompt.user_prompt,
                call_params,
            )
        except Exception as err:
            raise LLMGenerationError(
                f"LLM inference error ({request.model_identifier}): {err}",
                component_identifier=request.model_identifier,
                details={"error": str(err)},
            )

        return GenerationResult(
            model_identifier=request.model_identifier,
            raw_response=raw_text,
            generation_metadata={
                "component_version": self.component_version,
                "temperature": request.generation_parameters.temperature,
                "seed": request.generation_parameters.seed,
                "thinking_mode": request.thinking_mode,
            },
            usage_metadata={},
            finish_status=FinishStatus.COMPLETED,
        )
