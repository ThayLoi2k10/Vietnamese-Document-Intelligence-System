"""Context policy definitions and strategy enumerations."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class DeduplicationPolicy(str, Enum):
    DISABLED = "DISABLED"
    EXACT_CHUNK_ID = "EXACT_CHUNK_ID"
    EXACT_TEXT = "EXACT_TEXT"
    OVERLAP_THRESHOLD = "OVERLAP_THRESHOLD"
    SAME_ARTICLE_LIMIT = "SAME_ARTICLE_LIMIT"


class ExpansionPolicy(str, Enum):
    DISABLED = "DISABLED"
    PARENT_ARTICLE = "PARENT_ARTICLE"
    NEIGHBORING_CHUNKS = "NEIGHBORING_CHUNKS"
    ARTICLE_CONTEXT_WINDOW = "ARTICLE_CONTEXT_WINDOW"


class OrderingStrategy(str, Enum):
    SCORE_DESCENDING = "SCORE_DESCENDING"
    LEGAL_ORDER = "LEGAL_ORDER"


class BudgetOverflowPolicy(str, Enum):
    TRUNCATE_BY_POLICY = "TRUNCATE_BY_POLICY"
    DROP_LOWEST_PRIORITY_ITEMS = "DROP_LOWEST_PRIORITY_ITEMS"
    REQUIRE_SMALLER_CONTEXT_K = "REQUIRE_SMALLER_CONTEXT_K"
    RAISE_CONTEXT_BUDGET_ERROR = "RAISE_CONTEXT_BUDGET_ERROR"


@dataclass
class ContextPolicy:
    context_top_k: int = 5
    score_threshold: Optional[float] = None
    max_context_items: Optional[int] = None
    deduplication_policy: DeduplicationPolicy = DeduplicationPolicy.EXACT_CHUNK_ID
    deduplication_options: Dict[str, Any] = field(default_factory=dict)
    expansion_policy: ExpansionPolicy = ExpansionPolicy.DISABLED
    expansion_options: Dict[str, Any] = field(default_factory=dict)
    ordering_strategy: OrderingStrategy = OrderingStrategy.SCORE_DESCENDING
    max_context_tokens: int = 4096
    budget_overflow_policy: BudgetOverflowPolicy = (
        BudgetOverflowPolicy.DROP_LOWEST_PRIORITY_ITEMS
    )
    component_version: str = "context-policy-v1.0"
