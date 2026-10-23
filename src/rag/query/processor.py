"""Unified query processor coordinating normalization, rewriting, and fallbacks."""

import time
from typing import List, Optional

from src.rag.query.models import (
    FailureType,
    ProcessedQuery,
    ProcessingMetadata,
    ProcessingStatus,
    RewritingResult,
)
from src.rag.query.normalizer import LegalAcronymNormalizer
from src.rag.query.rewriter import LLMQueryRewriter, tokenize_words


class QueryProcessor:
    """End-to-end query processing pipeline for Vietnamese legal intelligence."""

    def __init__(
        self,
        normalizer: Optional[LegalAcronymNormalizer] = None,
        rewriter: Optional[LLMQueryRewriter] = None,
        enable_llm: bool = True,
        version: str = "v1.0.0",
    ) -> None:
        self.normalizer = normalizer or LegalAcronymNormalizer()
        self.rewriter = rewriter
        self.enable_llm = enable_llm
        self.version = version

    def process(self, query: str) -> ProcessedQuery:
        if not query or not query.strip():
            return ProcessedQuery(
                original_query=query,
                normalized_query=query,
                final_query=query,
                status=ProcessingStatus.UNCHANGED,
                metadata=ProcessingMetadata(processor_version=self.version),
            )

        clean_query = query.strip()
        input_word_count = len(tokenize_words(clean_query))

        t0 = time.perf_counter()
        normalized_query, acronyms_detected, rules_applied = self.normalizer.normalize(clean_query)
        t1 = time.perf_counter()
        normalizer_latency_ms = (t1 - t0) * 1000.0

        final_query = normalized_query
        status = ProcessingStatus.UNCHANGED
        failure_type: Optional[FailureType] = None
        statutory_terms: List[str] = []
        llm_latency_ms = 0.0

        if acronyms_detected and not self.enable_llm:
            status = ProcessingStatus.SUCCESS

        if self.enable_llm and self.rewriter is not None:
            t2 = time.perf_counter()
            rewriter_result: RewritingResult = self.rewriter.rewrite(normalized_query)
            t3 = time.perf_counter()
            llm_latency_ms = (t3 - t2) * 1000.0

            if rewriter_result.is_valid:
                statutory_terms = rewriter_result.statutory_terms
                expanded = rewriter_result.expanded_query.strip()

                if expanded.lower() == normalized_query.lower():
                    if acronyms_detected:
                        status = ProcessingStatus.SUCCESS
                        final_query = normalized_query
                    else:
                        status = ProcessingStatus.UNCHANGED
                        final_query = clean_query
                        failure_type = FailureType.INEFFECTIVE_UNCHANGED
                else:
                    status = ProcessingStatus.SUCCESS
                    final_query = expanded
                    rules_applied.append(f"LLM_EXPANSION:{len(statutory_terms)}_terms")
            else:
                err_msg = rewriter_result.error_message or ""
                if "Information loss" in err_msg:
                    failure_type = FailureType.INFORMATION_LOSS
                elif "JSON" in err_msg or "empty" in err_msg or "short" in err_msg:
                    failure_type = FailureType.MALFORMED_OUTPUT
                else:
                    failure_type = FailureType.QUERY_DRIFT

                if acronyms_detected:
                    status = ProcessingStatus.FALLBACK_NORMALIZED
                    final_query = normalized_query
                else:
                    status = ProcessingStatus.FALLBACK_RAW
                    final_query = clean_query

        total_latency_ms = normalizer_latency_ms + llm_latency_ms
        output_word_count = len(tokenize_words(final_query))

        metadata = ProcessingMetadata(
            processor_version=self.version,
            normalizer_latency_ms=round(normalizer_latency_ms, 3),
            llm_latency_ms=round(llm_latency_ms, 3),
            total_latency_ms=round(total_latency_ms, 3),
            model_identifier=self.rewriter.model_name if self.rewriter else None,
            prompt_id=self.rewriter.prompt_id if self.rewriter else None,
            input_word_count=input_word_count,
            output_word_count=output_word_count,
            acronyms_detected=acronyms_detected,
            rules_applied=rules_applied,
        )

        return ProcessedQuery(
            original_query=clean_query,
            normalized_query=normalized_query,
            final_query=final_query,
            status=status,
            failure_type=failure_type,
            acronyms_expanded=acronyms_detected,
            statutory_terms=statutory_terms,
            metadata=metadata,
        )
