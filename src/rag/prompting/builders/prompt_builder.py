"""Deterministic prompt builder assembling system and user prompts."""

from typing import Dict, Optional

from src.rag.context.domain.models import AssembledContext
from src.rag.errors.exceptions import InvalidPromptError
from src.rag.prompting.domain.models import PromptRequest, PromptTemplate
from src.rag.prompting.templates.registry import PromptTemplateRegistry


class PromptBuilder:
    """Assembles deterministic system and user prompts from context and template."""

    def __init__(
        self,
        template_registry: Optional[PromptTemplateRegistry] = None,
        default_version: str = "prompt_v1",
    ) -> None:
        self.template_registry = template_registry or PromptTemplateRegistry()
        self.default_version = default_version

    def build(
        self,
        query: str,
        context: AssembledContext,
        template_version: Optional[str] = None,
    ) -> PromptRequest:
        if not query or not query.strip():
            raise InvalidPromptError("Query cannot be empty.")

        version = template_version or self.default_version
        template: PromptTemplate = self.template_registry.get(version)

        evidence_blocks = []
        reference_mapping: Dict[str, str] = {}

        for item in context.items:
            ref_id = item.context_item_id
            reference_mapping[ref_id] = item.source_identity

            header = f"TÀI LIỆU {ref_id}: Văn bản {item.law_id}"
            if item.article_identifier:
                header += f", {item.article_identifier}"

            block = f"{header}\nNội dung:\n{item.text.strip()}"
            evidence_blocks.append(block)

        evidence_text = (
            "\n\n".join(evidence_blocks)
            if evidence_blocks
            else "(Không tìm thấy tài liệu căn cứ pháp lý phù hợp)"
        )

        try:
            rendered_user = template.user_template.format(
                query=query.strip(),
                evidence_text=evidence_text,
            )
            rendered_system = template.system_template
        except KeyError as err:
            raise InvalidPromptError(
                f"Missing template placeholder in '{version}': {err}"
            )

        return PromptRequest(
            system_prompt=rendered_system,
            user_prompt=rendered_user,
            prompt_version=version,
            evidence_registry=context.evidence_registry,
            prompt_metadata={
                "template_name": template.template_name,
                "instruction_policy_version": template.instruction_policy_version,
                "evidence_count": len(context.items),
                "context_tokens": context.total_tokens,
                "reference_mapping": reference_mapping,
            },
        )
