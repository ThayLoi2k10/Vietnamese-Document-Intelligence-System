"""Domain models for query processing, rewriting, and telemetry."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ProcessingStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FALLBACK_NORMALIZED = "FALLBACK_NORMALIZED"
    FALLBACK_RAW = "FALLBACK_RAW"
    UNCHANGED = "UNCHANGED"


class FailureType(str, Enum):
    QUERY_DRIFT = "QUERY_DRIFT"
    INFORMATION_LOSS = "INFORMATION_LOSS"
    HALLUCINATION_ADDITION = "HALLUCINATION_ADDITION"
    MALFORMED_OUTPUT = "MALFORMED_OUTPUT"
    INEFFECTIVE_UNCHANGED = "INEFFECTIVE_UNCHANGED"


@dataclass
class RewritingResult:
    intent: str = ""
    statutory_terms: List[str] = field(default_factory=list)
    expanded_query: str = ""
    confidence: str = "HIGH"
    is_valid: bool = True
    error_message: Optional[str] = None


@dataclass
class ProcessingMetadata:
    processor_version: str = "v1.0.0"
    normalizer_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    model_identifier: Optional[str] = None
    prompt_id: Optional[str] = None
    input_word_count: int = 0
    output_word_count: int = 0
    acronyms_detected: Dict[str, str] = field(default_factory=dict)
    rules_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessedQuery:
    original_query: str
    normalized_query: str
    final_query: str
    status: ProcessingStatus = ProcessingStatus.SUCCESS
    failure_type: Optional[FailureType] = None
    acronyms_expanded: Dict[str, str] = field(default_factory=dict)
    statutory_terms: List[str] = field(default_factory=list)
    metadata: ProcessingMetadata = field(default_factory=ProcessingMetadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "final_query": self.final_query,
            "status": self.status.value,
            "failure_type": self.failure_type.value if self.failure_type else None,
            "acronyms_expanded": self.acronyms_expanded,
            "statutory_terms": self.statutory_terms,
            "metadata": self.metadata.to_dict(),
        }
