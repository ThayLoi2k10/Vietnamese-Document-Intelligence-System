# src/rag/abstention/policies/abstention_policy.py
"""Chính sách từ chối trả lời (AbstentionPolicy).

Tuân thủ Mục 15.1, 15.4: Chính sách quyết định và kiểm định mức tin cậy.
"""

from dataclasses import dataclass


@dataclass
class AbstentionPolicy:
    """Cấu hình các ngưỡng và quy tắc từ chối trả lời."""

    min_context_items_for_full: int = 1
    min_valid_citations_for_full: int = 1
    fallback_to_abstention_on_empty_retrieval: bool = True
    downgrade_on_invalid_citations: bool = True
    policy_version: str = "abstention-policy-v1.0"
