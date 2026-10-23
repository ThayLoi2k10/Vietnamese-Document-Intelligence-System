"""Context assembler coordinating filtering, deduplication, expansion, ordering, and token budget."""

from typing import Any, Dict, List, Optional, Set, Union

from src.rag.context.domain.models import AssembledContext, ContextItem
from src.rag.context.domain.token_counter import (
    CharacterTokenCounter,
    TokenCounter,
)
from src.rag.context.policies.policy_types import (
    BudgetOverflowPolicy,
    ContextPolicy,
    DeduplicationPolicy,
    ExpansionPolicy,
    OrderingStrategy,
)
from src.rag.errors.exceptions import ContextBudgetExceededError
from src.rag.reranking.domain.models import RankedCandidate, RankedResults
from src.rag.retrieval.domain.models import RetrievalItem, RetrievalResult


class ContextAssembler:
    """Assembles retrieved or reranked evidence into a budget-enforced, ordered context."""

    def __init__(
        self,
        default_policy: Optional[ContextPolicy] = None,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        self.default_policy = default_policy or ContextPolicy()
        self.token_counter = token_counter or CharacterTokenCounter()

    def _extract_candidates(
        self,
        ranked_evidence: Union[
            RetrievalResult,
            RankedResults,
            List[RetrievalItem],
            List[RankedCandidate],
        ],
    ) -> List[Dict[str, Any]]:
        extracted: List[Dict[str, Any]] = []

        if isinstance(ranked_evidence, RetrievalResult):
            for item in ranked_evidence.items:
                extracted.append(
                    {
                        "item": item,
                        "score": item.score,
                        "rank": item.rank,
                    }
                )
        elif isinstance(ranked_evidence, RankedResults):
            for candidate in ranked_evidence.candidates:
                extracted.append(
                    {
                        "item": candidate.retrieval_item,
                        "score": (
                            candidate.rerank_score
                            if candidate.rerank_score is not None
                            else candidate.retrieval_score
                        ),
                        "rank": candidate.rerank_rank,
                    }
                )
        elif isinstance(ranked_evidence, list):
            for elem in ranked_evidence:
                if isinstance(elem, RankedCandidate):
                    extracted.append(
                        {
                            "item": elem.retrieval_item,
                            "score": (
                                elem.rerank_score
                                if elem.rerank_score is not None
                                else elem.retrieval_score
                            ),
                            "rank": elem.rerank_rank,
                        }
                    )
                elif isinstance(elem, RetrievalItem):
                    extracted.append(
                        {
                            "item": elem,
                            "score": elem.score,
                            "rank": elem.rank,
                        }
                    )
        return extracted

    def assemble(
        self,
        ranked_evidence: Union[
            RetrievalResult,
            RankedResults,
            List[RetrievalItem],
            List[RankedCandidate],
        ],
        policy: Optional[ContextPolicy] = None,
        query: str = "",
        token_counter: Optional[TokenCounter] = None,
    ) -> AssembledContext:
        active_policy = policy or self.default_policy
        counter = token_counter or self.token_counter

        candidates = self._extract_candidates(ranked_evidence)

        if active_policy.score_threshold is not None:
            candidates = [
                c for c in candidates
                if c["score"] is None or c["score"] >= active_policy.score_threshold
            ]

        deduped_candidates = self._deduplicate(
            candidates,
            active_policy.deduplication_policy,
            active_policy.deduplication_options,
        )

        limit = active_policy.context_top_k
        if active_policy.max_context_items is not None:
            limit = min(limit, active_policy.max_context_items)
        selected_candidates = deduped_candidates[:limit]

        expanded_candidates = self._expand(
            selected_candidates,
            active_policy.expansion_policy,
            active_policy.expansion_options,
        )

        ordered_candidates = self._order(
            expanded_candidates, active_policy.ordering_strategy
        )

        final_items, total_tokens = self._enforce_token_budget(
            ordered_candidates, active_policy, counter
        )

        evidence_registry: Dict[str, ContextItem] = {}
        for it in final_items:
            evidence_registry[it.context_item_id] = it
            bare_id = it.context_item_id.strip("[]")
            if bare_id not in evidence_registry:
                evidence_registry[bare_id] = it

        return AssembledContext(
            query=query,
            items=final_items,
            total_tokens=total_tokens,
            evidence_registry=evidence_registry,
            context_metadata={
                "component_version": active_policy.component_version,
                "input_candidates": len(candidates),
                "after_dedup": len(deduped_candidates),
                "final_items": len(final_items),
                "total_tokens": total_tokens,
                "policy": {
                    "context_top_k": active_policy.context_top_k,
                    "deduplication_policy": active_policy.deduplication_policy.value,
                    "expansion_policy": active_policy.expansion_policy.value,
                    "ordering_strategy": active_policy.ordering_strategy.value,
                    "max_context_tokens": active_policy.max_context_tokens,
                },
            },
        )

    def _deduplicate(
        self,
        candidates: List[Dict[str, Any]],
        policy: DeduplicationPolicy,
        options: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if policy == DeduplicationPolicy.DISABLED:
            return candidates

        result: List[Dict[str, Any]] = []
        seen_chunk_ids: Set[str] = set()
        seen_texts: Set[str] = set()
        article_counts: Dict[str, int] = {}
        max_per_article = options.get("max_chunks_per_article", 2)
        overlap_threshold = options.get("overlap_threshold", 0.8)

        for c in candidates:
            item: RetrievalItem = c["item"]

            if policy == DeduplicationPolicy.EXACT_CHUNK_ID:
                if item.chunk_id and item.chunk_id in seen_chunk_ids:
                    continue
                if item.chunk_id:
                    seen_chunk_ids.add(item.chunk_id)

            elif policy == DeduplicationPolicy.EXACT_TEXT:
                norm_text = item.text.strip().lower()
                if norm_text in seen_texts:
                    continue
                seen_texts.add(norm_text)

            elif policy == DeduplicationPolicy.SAME_ARTICLE_LIMIT:
                art_key = (
                    str(item.global_article_id)
                    if item.global_article_id is not None
                    else f"{item.law_id}_{item.article_identifier}"
                )
                curr_cnt = article_counts.get(art_key, 0)
                if curr_cnt >= max_per_article:
                    continue
                article_counts[art_key] = curr_cnt + 1

            elif policy == DeduplicationPolicy.OVERLAP_THRESHOLD:
                is_overlap = False
                c_words = set(item.text.lower().split())
                for accepted in result:
                    a_words = set(accepted["item"].text.lower().split())
                    if not c_words or not a_words:
                        continue
                    jaccard = len(c_words & a_words) / len(c_words | a_words)
                    if jaccard >= overlap_threshold:
                        is_overlap = True
                        break
                if is_overlap:
                    continue

            result.append(c)
        return result

    def _expand(
        self,
        candidates: List[Dict[str, Any]],
        policy: ExpansionPolicy,
        options: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if policy == ExpansionPolicy.DISABLED:
            return candidates

        expanded: List[Dict[str, Any]] = []
        for c in candidates:
            item: RetrievalItem = c["item"]
            expanded_text = item.text
            if policy == ExpansionPolicy.PARENT_ARTICLE:
                parent_text = item.source_metadata.get("parent_text")
                if parent_text:
                    expanded_text = parent_text
            elif policy == ExpansionPolicy.NEIGHBORING_CHUNKS:
                prev_text = item.source_metadata.get("prev_chunk_text", "")
                next_text = item.source_metadata.get("next_chunk_text", "")
                parts = [p for p in [prev_text, item.text, next_text] if p]
                expanded_text = "\n".join(parts)

            c_expanded = dict(c)
            if expanded_text != item.text:
                c_expanded["item"] = RetrievalItem(
                    chunk_id=item.chunk_id,
                    global_article_id=item.global_article_id,
                    law_id=item.law_id,
                    article_identifier=item.article_identifier,
                    text=expanded_text,
                    chunk_order=item.chunk_order,
                    total_chunks=item.total_chunks,
                    score=item.score,
                    rank=item.rank,
                    token_count=item.token_count,
                    parent_reference=item.parent_reference,
                    source_metadata=dict(item.source_metadata),
                )
            expanded.append(c_expanded)
        return expanded

    def _order(
        self,
        candidates: List[Dict[str, Any]],
        strategy: OrderingStrategy,
    ) -> List[Dict[str, Any]]:
        if strategy == OrderingStrategy.SCORE_DESCENDING:
            return sorted(
                candidates,
                key=lambda x: (
                    x["score"] if x["score"] is not None else -1e9
                ),
                reverse=True,
            )
        elif strategy == OrderingStrategy.LEGAL_ORDER:
            return sorted(
                candidates,
                key=lambda x: (
                    x["item"].law_id,
                    x["item"].article_identifier or "",
                    x["item"].chunk_order or 0,
                ),
            )
        return candidates

    def _enforce_token_budget(
        self,
        candidates: List[Dict[str, Any]],
        policy: ContextPolicy,
        counter: TokenCounter,
    ) -> tuple[List[ContextItem], int]:
        max_tokens = policy.max_context_tokens
        overflow_action = policy.budget_overflow_policy

        accepted_items: List[ContextItem] = []
        current_tokens = 0

        for c in candidates:
            item: RetrievalItem = c["item"]
            item_text = item.text
            item_tokens = (
                item.token_count
                if item.token_count is not None
                else counter.count_tokens(item_text)
            )

            if current_tokens + item_tokens <= max_tokens:
                ctx_item = ContextItem(
                    context_item_id=f"[{len(accepted_items) + 1}]",
                    source_identity=item.stable_identifier,
                    law_id=item.law_id,
                    article_identifier=item.article_identifier,
                    global_article_id=item.global_article_id,
                    chunk_id=item.chunk_id,
                    text=item_text,
                    source_rank=c["rank"],
                    source_score=c["score"],
                    context_position=len(accepted_items) + 1,
                    token_count=item_tokens,
                    internal_metadata=item.source_metadata,
                )
                accepted_items.append(ctx_item)
                current_tokens += item_tokens
            else:
                if overflow_action == BudgetOverflowPolicy.RAISE_CONTEXT_BUDGET_ERROR:
                    raise ContextBudgetExceededError(
                        f"Context size exceeded token budget: {current_tokens + item_tokens} > {max_tokens}."
                    )
                elif overflow_action == BudgetOverflowPolicy.REQUIRE_SMALLER_CONTEXT_K:
                    raise ContextBudgetExceededError(
                        f"Token budget insufficient for context_top_k={policy.context_top_k}."
                    )
                elif overflow_action == BudgetOverflowPolicy.DROP_LOWEST_PRIORITY_ITEMS:
                    continue
                elif overflow_action == BudgetOverflowPolicy.TRUNCATE_BY_POLICY:
                    remaining_tokens = max(0, max_tokens - current_tokens)
                    if remaining_tokens > 10:
                        chars_to_keep = int(remaining_tokens * 3.5)
                        truncated_text = item_text[:chars_to_keep] + "... [đã cắt bớt]"
                        t_tokens = counter.count_tokens(truncated_text)
                        ctx_item = ContextItem(
                            context_item_id=f"[{len(accepted_items) + 1}]",
                            source_identity=item.stable_identifier,
                            law_id=item.law_id,
                            article_identifier=item.article_identifier,
                            global_article_id=item.global_article_id,
                            chunk_id=item.chunk_id,
                            text=truncated_text,
                            source_rank=c["rank"],
                            source_score=c["score"],
                            context_position=len(accepted_items) + 1,
                            token_count=t_tokens,
                            internal_metadata=item.source_metadata,
                        )
                        accepted_items.append(ctx_item)
                        current_tokens += t_tokens
                    break

        return accepted_items, current_tokens
