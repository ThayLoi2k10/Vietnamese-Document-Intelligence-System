"""Độ đo đánh giá truy xuất thông tin: Precision, Recall, MRR, MAP, nDCG."""

import math
from typing import Sequence


def compute_precision_at_k(
    retrieved_ids: list[int],
    relevant_ids: list[int] | set[int],
    k: int,
) -> float:
    if k <= 0 or not retrieved_ids:
        return 0.0

    rel_set = set(relevant_ids)
    cutoff = retrieved_ids[:k]
    seen = set()
    hits = 0

    for doc_id in cutoff:
        if doc_id in rel_set and doc_id not in seen:
            hits += 1
            seen.add(doc_id)

    return hits / k


def compute_recall_at_k(
    retrieved_ids: list[int],
    relevant_ids: list[int] | set[int],
    k: int,
) -> float:
    if k <= 0 or not retrieved_ids:
        return 0.0

    rel_set = set(relevant_ids)
    if not rel_set:
        return 0.0

    cutoff = retrieved_ids[:k]
    seen = set()
    hits = 0

    for doc_id in cutoff:
        if doc_id in rel_set and doc_id not in seen:
            hits += 1
            seen.add(doc_id)

    return hits / len(rel_set)


def compute_mrr_at_k(
    retrieved_ids: list[int],
    relevant_ids: list[int] | set[int],
    k: int,
) -> float:
    if k <= 0 or not retrieved_ids:
        return 0.0

    rel_set = set(relevant_ids)
    cutoff = retrieved_ids[:k]
    for rank_idx, doc_id in enumerate(cutoff, start=1):
        if doc_id in rel_set:
            return 1.0 / rank_idx
    return 0.0


def compute_ap_at_k(
    retrieved_ids: list[int],
    relevant_ids: list[int] | set[int],
    k: int,
) -> float:
    if k <= 0 or not retrieved_ids:
        return 0.0

    rel_set = set(relevant_ids)
    if not rel_set:
        return 0.0

    cutoff = retrieved_ids[:k]
    hits = 0
    sum_precisions = 0.0
    seen = set()

    for rank_idx, doc_id in enumerate(cutoff, start=1):
        if doc_id in rel_set and doc_id not in seen:
            hits += 1
            sum_precisions += hits / rank_idx
            seen.add(doc_id)

    normalizer = min(len(rel_set), k)
    return sum_precisions / normalizer if normalizer > 0 else 0.0


def compute_ndcg_at_k(
    retrieved_ids: list[int],
    relevant_ids: list[int] | set[int],
    k: int,
) -> float:
    if k <= 0 or not retrieved_ids:
        return 0.0

    rel_set = set(relevant_ids)
    if not rel_set:
        return 0.0

    cutoff = retrieved_ids[:k]
    dcg = 0.0
    seen = set()

    for rank_idx, doc_id in enumerate(cutoff, start=1):
        if doc_id in rel_set and doc_id not in seen:
            dcg += 1.0 / math.log2(rank_idx + 1)
            seen.add(doc_id)

    ideal_hits = min(len(rel_set), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_query_retrieval(
    retrieved_ids: list[int],
    relevant_ids: list[int] | set[int],
    k_list: Sequence[int] = (1, 5, 10, 20, 50, 100),
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    for k in k_list:
        metrics[f"P@{k}"] = compute_precision_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"R@{k}"] = compute_recall_at_k(retrieved_ids, relevant_ids, k)

    metrics["MRR@10"] = compute_mrr_at_k(retrieved_ids, relevant_ids, k=10)
    metrics["MAP@10"] = compute_ap_at_k(retrieved_ids, relevant_ids, k=10)
    metrics["MAP@100"] = compute_ap_at_k(retrieved_ids, relevant_ids, k=100)
    metrics["nDCG@10"] = compute_ndcg_at_k(retrieved_ids, relevant_ids, k=10)
    metrics["nDCG@100"] = compute_ndcg_at_k(retrieved_ids, relevant_ids, k=100)

    return metrics


def aggregate_retrieval_metrics(
    per_query_metrics: list[dict[str, float]]
) -> dict[str, float]:
    if not per_query_metrics:
        return {}

    metric_names = per_query_metrics[0].keys()
    num_queries = len(per_query_metrics)

    return {
        name: sum(q.get(name, 0.0) for q in per_query_metrics) / num_queries
        for name in metric_names
    }