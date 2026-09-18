# src/rag/prompting/domain/models.py
"""Mô hình miền cho hệ thống Prompting Phase 9 RAG.

Tuân thủ Mục 11: Prompt Architecture:
    - 11.1 Components: PromptTemplate, PromptContext, GenerationInstructions, PromptBuilder
    - 11.2 Versioned PromptTemplate
    - 11.3 GenerationInstructions (Grounding, Citation, Abstention, Format)
    - 11.4 PromptContext
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.rag.context.domain.models import ContextItem


@dataclass
class GenerationInstructions:
    """Đặc tả các nhóm chỉ dẫn cho mô hình ngôn ngữ lớn."""

    grounding_instruction: str
    citation_instruction: str
    abstention_instruction: str
    formatting_instruction: str
    policy_version: str = "instructions-v1.0"

    def to_full_instruction_text(self) -> str:
        return (
            f"1. NGUYÊN TẮC CĂN CỨ PHÁP LÝ (GROUNDING):\n{self.grounding_instruction}\n\n"
            f"2. NGUYÊN TẮC TRÍCH DẪN (CITATION):\n{self.citation_instruction}\n\n"
            f"3. NGUYÊN TẮC TỪ CHỐI TRẢ LỜI (ABSTENTION):\n{self.abstention_instruction}\n\n"
            f"4. ĐỊNH DẠNG ĐẦU RA (OUTPUT FORMAT):\n{self.formatting_instruction}"
        )


@dataclass
class PromptTemplate:
    """Template định nghĩa cấu trúc prompt có phiên bản (Versioned Prompt Template)."""

    template_version: str  # Ví dụ: "prompt_v1"
    template_name: str
    system_template: str
    user_template: str
    instruction_policy_version: str = "v1.0"
    created_at: str = "2026-09-18"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PromptContext:
    """Tập hợp dữ liệu cần thiết trước khi xây dựng prompt."""

    query: str
    evidence_items: List[Dict[str, Any]]
    citation_reference_mapping: Dict[str, str]
    prompt_version: str
    system_instructions: Optional[str] = None
    extra_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptRequest:
    """Prompt hoàn chỉnh đã được render sẵn sàng gửi tới LLM Generator."""

    system_prompt: str
    user_prompt: str
    prompt_version: str
    evidence_registry: Dict[str, ContextItem]
    prompt_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "prompt_version": self.prompt_version,
            "prompt_metadata": self.prompt_metadata,
        }
