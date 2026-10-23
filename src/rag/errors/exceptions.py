"""Explicit error taxonomy and custom exceptions for the RAG pipeline."""

from typing import Any, Dict, Optional


class RAGError(Exception):
    """Base exception for all RAG pipeline errors with telemetry support."""

    default_stage: Optional[str] = None

    def __init__(
        self,
        message: str,
        experiment_id: Optional[str] = None,
        pipeline_stage: Optional[str] = None,
        component_identifier: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.experiment_id = experiment_id
        self.pipeline_stage = pipeline_stage or self.default_stage
        self.component_identifier = component_identifier
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "experiment_id": self.experiment_id,
            "pipeline_stage": self.pipeline_stage,
            "component_identifier": self.component_identifier,
            "details": self.details,
        }


class InvalidRetrievalResultError(RAGError):
    default_stage = "retrieval"


class MissingRequiredMetadataError(RAGError):
    default_stage = None


class EmptyRetrievalError(RAGError):
    default_stage = "retrieval"


class ContextBudgetExceededError(RAGError):
    default_stage = "context_assembly"


class InvalidPromptError(RAGError):
    default_stage = "prompting"


class LLMGenerationError(RAGError):
    default_stage = "generation"


class MalformedStructuredOutputError(RAGError):
    default_stage = "output_parsing"


class InvalidCitationError(RAGError):
    default_stage = "citation_validation"


class UnsupportedModelConfigurationError(RAGError):
    default_stage = "generation"


class EvaluationDataLeakageError(RAGError):
    default_stage = "dataset_protection"


class IncompatibleComponentVersionError(RAGError):
    default_stage = None


class ConfigurationValidationError(RAGError):
    default_stage = "configuration"
