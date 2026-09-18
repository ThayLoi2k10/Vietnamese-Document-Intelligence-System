# src/rag/generation/domain/models.py
"""Mô hình miền cho hệ thống sinh văn bản LLM Generation Phase 9.

Tuân thủ Mục 12: LLM Generator Interface:
    - 12.3 GenerationRequest
    - 12.4 GenerationResult
    - 12.5 Thinking mode
    - 12.6 Determinism
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.rag.prompting.domain.models import PromptRequest


class FinishStatus(str, Enum):
    COMPLETED = "COMPLETED"
    MAX_TOKENS = "MAX_TOKENS"
    STOP_SEQUENCE = "STOP_SEQUENCE"
    ERROR = "ERROR"
    FILTERED = "FILTERED"


@dataclass
class GenerationParameters:
    """Các siêu tham số điều khiển quá trình sinh của mô hình ngôn ngữ."""

    temperature: float = 0.0
    top_p: float = 1.0
    seed: Optional[int] = None
    stop_sequences: List[str] = field(default_factory=list)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationRequest:
    """Yêu cầu suy luận gửi tới LLM Generator."""

    model_identifier: str
    prompt: PromptRequest
    max_output_tokens: int = 1024
    generation_parameters: GenerationParameters = field(
        default_factory=GenerationParameters
    )
    thinking_mode: bool = False
    reproducibility_options: Dict[str, Any] = field(default_factory=dict)
    request_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_identifier": self.model_identifier,
            "prompt": self.prompt.to_dict(),
            "max_output_tokens": self.max_output_tokens,
            "generation_parameters": self.generation_parameters.to_dict(),
            "thinking_mode": self.thinking_mode,
            "reproducibility_options": self.reproducibility_options,
            "request_metadata": self.request_metadata,
        }


@dataclass
class GenerationResult:
    """Kết quả sinh thô và metadata từ LLM Generator."""

    model_identifier: str
    raw_response: str
    generation_metadata: Dict[str, Any] = field(default_factory=dict)
    usage_metadata: Dict[str, int] = field(default_factory=dict)
    finish_status: FinishStatus = FinishStatus.COMPLETED
    parsed_response: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_identifier": self.model_identifier,
            "raw_response": self.raw_response,
            "finish_status": self.finish_status.value,
            "usage_metadata": self.usage_metadata,
            "generation_metadata": self.generation_metadata,
        }
