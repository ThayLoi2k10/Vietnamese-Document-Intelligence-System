"""Policy configuration parameters for abstention thresholds."""

from dataclasses import dataclass


@dataclass
class AbstentionPolicy:
    min_context_items_for_full: int = 1
    min_valid_citations_for_full: int = 1
    fallback_to_abstention_on_empty_retrieval: bool = True
    downgrade_on_invalid_citations: bool = True
    policy_version: str = "abstention-policy-v1.0"
