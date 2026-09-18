# src/rag/prompting/builders/prompt_builder.py
"""Xây dựng PromptRequest hoàn chỉnh từ AssembledContext và câu hỏi người dùng.

Tuân thủ Mục 11.5: Prompt builder responsibilities & Mục 7.4: Internal vs public metadata.
"""

from typing import Dict, Optional

from src.rag.context.domain.models import AssembledContext, ContextItem
from src.rag.errors.exceptions import InvalidPromptError
from src.rag.prompting.domain.models import PromptRequest, PromptTemplate
from src.rag.prompting.templates.registry import PromptTemplateRegistry


class PromptBuilder:
    """Builder chịu trách nhiệm lắp ghép prompt tất định từ context và template."""

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
        """Tạo PromptRequest tất định."""
        if not query or not query.strip():
            raise InvalidPromptError("Câu hỏi truy vấn không được để trống.")

        version = template_version or self.default_version
        template: PromptTemplate = self.template_registry.get(version)

        # Định dạng danh sách bằng chứng pháp lý đưa vào prompt (Chỉ dữ liệu công khai theo Mục 7.4)
        evidence_blocks = []
        reference_mapping: Dict[str, str] = {}

        for idx, item in enumerate(context.items, start=1):
            ref_id = item.context_item_id  # Ví dụ: "[1]"
            reference_mapping[ref_id] = item.source_identity

            header = f"TÀI LIỆU {ref_id}: Văn bản {item.law_id}"
            if item.article_identifier:
                header += f", {item.article_identifier}"

            block = f"{header}\nNội dung:\n{item.text.strip()}"
            evidence_blocks.append(block)

        if evidence_blocks:
            evidence_text = "\n\n".join(evidence_blocks)
        else:
            evidence_text = "(Không tìm thấy tài liệu căn cứ pháp lý phù hợp)"

        # Render template
        try:
            rendered_user = template.user_template.format(
                query=query.strip(),
                evidence_text=evidence_text,
            )
            rendered_system = template.system_template
        except KeyError as e:
            raise InvalidPromptError(
                f"Thiếu trường thay thế khi render prompt template '{version}': {e}"
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
