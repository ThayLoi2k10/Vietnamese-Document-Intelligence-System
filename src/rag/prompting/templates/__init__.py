# src/rag/prompting/templates/__init__.py
from src.rag.prompting.templates.registry import (
    DEFAULT_SYSTEM_TEMPLATE_V1,
    DEFAULT_USER_TEMPLATE_V1,
    PromptTemplateRegistry,
)

__all__ = [
    "PromptTemplateRegistry",
    "DEFAULT_SYSTEM_TEMPLATE_V1",
    "DEFAULT_USER_TEMPLATE_V1",
]
