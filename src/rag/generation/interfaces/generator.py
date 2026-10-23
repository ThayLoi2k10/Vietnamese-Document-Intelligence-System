"""Abstract LLM generator interface for language model inference."""

from abc import ABC, abstractmethod

from src.rag.generation.domain.models import GenerationRequest, GenerationResult


class LLMGenerator(ABC):
    """Abstract interface for LLM inference components."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        pass
