"""Domain models for generation instructions, templates, and prompt requests."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.rag.context.domain.models import ContextItem


@dataclass
class GenerationInstructions:
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
    template_version: str
    template_name: str
    system_template: str
    user_template: str
    instruction_policy_version: str = "v1.0"
    created_at: str = "2026-09-18"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PromptContext:
    query: str
    evidence_items: List[Dict[str, Any]]
    citation_reference_mapping: Dict[str, str]
    prompt_version: str
    system_instructions: Optional[str] = None
    extra_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptRequest:
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
