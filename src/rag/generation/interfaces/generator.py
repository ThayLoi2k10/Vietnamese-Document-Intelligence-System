# src/rag/generation/interfaces/generator.py
"""Interface trừu tượng cho LLM Generator.

Tuân thủ Mục 12: LLM Generator Interface:
    - 12.1 Interface: generate(request: GenerationRequest) -> GenerationResult
    - 12.2 Model abstraction: Không hard-code model name
"""

from abc import ABC, abstractmethod

from src.rag.generation.domain.models import GenerationRequest, GenerationResult


class LLMGenerator(ABC):
    """Cổng giao tiếp trừu tượng cho việc suy luận LLM."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Thực thi suy luận văn bản từ yêu cầu.

        Args:
            request: Đối tượng GenerationRequest chứa prompt, tham số và metadata.

        Returns:
            GenerationResult chứa kết quả thô, usage và trạng thái hoàn thành.
        """
        pass
