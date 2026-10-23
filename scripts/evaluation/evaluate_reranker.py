"""Evaluate cross-encoder reranker on hybrid retrieval candidates."""

from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FlagEmbedding import FlagReranker

from src.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    evaluate_query_retrieval,
)


def atomic_write_json(file_path: Path, data: Any) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=file_path.parent, delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp_name = tmp.name
    os.replace(tmp_name, file_path)


def load_corpus_dict(corpus_path: Path) -> Dict[int, str]:
    print(f"Loading corpus from {corpus_path}...")
    df = pd.read_parquet(
        corpus_path, columns=["global_article_id", "text_preprocessed"]
    )
    corpus_dict = dict(zip(df["global_article_id"], df["text_preprocessed"]))
    print(f"Loaded {len(corpus_dict)} articles.")
    return corpus_dict


def run_reranker_evaluation() -> None:
    print("=" * 80)
    print("PHASE 12: CROSS-ENCODER RERANKING EVALUATION")
    print("=" * 80)

    # 1. Paths
    hybrid_preds_path = PROJECT_ROOT / "results" / "benchmarks" / "hybrid_test_predictions.json"
    corpus_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "legal_retrieval"
        / "corpus_articles_preprocessed.parquet"
    )
    dense_metrics_path = (
        PROJECT_ROOT / "results" / "benchmarks" / "dense_test_metrics.json"
    )
    bm25_metrics_path = (
        PROJECT_ROOT / "results" / "benchmarks" / "bm25_tuned_test_metrics.json"
    )
    hybrid_metrics_path = (
        PROJECT_ROOT / "results" / "benchmarks" / "hybrid_test_metrics.json"
    )

    out_metrics_path = PROJECT_ROOT / "results" / "benchmarks" / "reranked_test_metrics.json"
    out_preds_path = PROJECT_ROOT / "results" / "benchmarks" / "reranked_test_predictions.json"
    out_comp_path = (
        PROJECT_ROOT / "results" / "benchmarks" / "reranker_vs_baselines_comparison.json"
    )
    out_scaling_path = (
        PROJECT_ROOT / "results" / "benchmarks" / "reranker_candidate_pool_scaling.json"
    )

    if not hybrid_preds_path.exists():
        raise FileNotFoundError(f"Missing {hybrid_preds_path}. Run evaluate_hybrid.py first.")

    with open(hybrid_preds_path, "r", encoding="utf-8") as f:
        hybrid_predictions = json.load(f)

    num_queries = len(hybrid_predictions)
    print(f"Loaded {num_queries} queries from {hybrid_preds_path}.")

    corpus_dict = load_corpus_dict(corpus_path)

    # 2. Initialize FlagReranker with optimal settings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_LENGTH = 256
    BATCH_SIZE = 32

    print(f"Initializing FlagReranker('BAAI/bge-reranker-base') on {device} (use_fp16={torch.cuda.is_available()})...")
    reranker = FlagReranker(
        "BAAI/bge-reranker-base",
        use_fp16=torch.cuda.is_available(),
        device=device,
        max_length=MAX_LENGTH,
    )
    print("Reranker loaded successfully.")

    # Warm-up
    print("Warming up GPU kernels...")
    warmup_pairs = [["Truy vấn kiểm tra?", "Nội dung điều luật văn bản pháp luật kiểm tra."]] * 32
    _ = reranker.compute_score(warmup_pairs, batch_size=BATCH_SIZE, max_length=MAX_LENGTH)
    print("Warm-up complete.")

    # 3. Prepare pairs for Top-20 candidates across all 438 queries
    POOL_20 = 20
    top20_pairs: List[Tuple[str, str]] = []
    query_meta_20: List[Dict[str, Any]] = []

    print(f"Preparing pairs for Top-{POOL_20} candidates across {num_queries} queries...")
    for q_idx, item in enumerate(hybrid_predictions):
        query_text = item["query_text"]
        top_100_ids = item["top_100_retrieved_ids"]
        pool_ids = top_100_ids[:POOL_20]
        start_idx = len(top20_pairs)
        for art_id in pool_ids:
            doc_text = corpus_dict.get(art_id, "")
            top20_pairs.append((query_text, doc_text))
        end_idx = len(top20_pairs)
        query_meta_20.append({
            "query_id": item["query_id"],
            "query_text": query_text,
            "ground_truth": item["ground_truth"],
            "top_100_ids": top_100_ids,
            "pool_ids": pool_ids,
            "start_idx": start_idx,
            "end_idx": end_idx,
        })

    print(f"Total Top-20 pairs to score: {len(top20_pairs)} ({len(top20_pairs)/num_queries:.1f} pairs/query)")

    # 4. Compute scores for Top-20 candidate pool (with disk caching)
    cache_top20_path = PROJECT_ROOT / "results" / "benchmarks" / "reranker_top20_scores_cache.json"
    if cache_top20_path.exists():
        print(f"Loading cached Top-20 scores from {cache_top20_path}...")
        with open(cache_top20_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        scores_20 = cache_data["scores"]
        time_20 = cache_data["time_sec"]
    else:
        print(f"Executing batch scoring (batch_size={BATCH_SIZE}, max_length={MAX_LENGTH})...")
        t0 = time.time()
        scores_raw_20 = reranker.compute_score(
            [[q, d] for q, d in top20_pairs],
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
        )
        time_20 = time.time() - t0
        scores_20 = [float(s) for s in scores_raw_20]
        atomic_write_json(cache_top20_path, {"scores": scores_20, "time_sec": time_20})
        print(f"Saved Top-20 scores cache to {cache_top20_path}")

    ms_per_pair = (time_20 * 1000) / len(top20_pairs)
    ms_per_query_top20 = (time_20 * 1000) / num_queries
    qps_top20 = num_queries / time_20
    print(f"Scored {len(top20_pairs)} pairs in {time_20:.2f}s!")
    print(f"  Speed: {ms_per_pair:.2f} ms/pair | Top-20 query latency: {ms_per_query_top20:.2f} ms | QPS: {qps_top20:.2f}")

    # 5. Evaluate Candidate Pools Top-10 and Top-20 across all 438 queries
    primary_eval_results: Dict[str, Any] = {}
    predictions_top20: List[Dict[str, Any]] = []

    for pool_size in [10, 20]:
        print(f"\n--- Full Benchmark (438 queries): Candidate Pool Top-{pool_size} ---")
        q_metrics = []
        preds = []

        for q_meta in query_meta_20:
            q_id = q_meta["query_id"]
            q_text = q_meta["query_text"]
            gt = q_meta["ground_truth"]
            top_100 = q_meta["top_100_ids"]
            start_i = q_meta["start_idx"]

            subpool_ids = top_100[:pool_size]
            subpool_scores = scores_20[start_i : start_i + pool_size]

            scored = sorted(zip(subpool_ids, subpool_scores), key=lambda x: x[1], reverse=True)
            reranked_ids = [c[0] for c in scored] + top_100[pool_size:]

            m = evaluate_query_retrieval(reranked_ids, gt)
            q_metrics.append(m)

            top_10_info = []
            for rank_idx, art_id in enumerate(reranked_ids[:10], start=1):
                sc = next((s for a, s in scored if a == art_id), -999.0)
                top_10_info.append({
                    "article_id": art_id,
                    "rerank_score": sc,
                    "rank": rank_idx,
                    "is_relevant": art_id in gt,
                })

            preds.append({
                "query_id": q_id,
                "query_text": q_text,
                "ground_truth": gt,
                "metrics": m,
                "top_10_retrieved": top_10_info,
                "top_100_retrieved_ids": reranked_ids,
            })

        macro_m = aggregate_retrieval_metrics(q_metrics)
        zero_hits_10 = sum(1 for m in q_metrics if m["R@10"] == 0.0)
        zero_hits_100 = sum(1 for m in q_metrics if m["R@100"] == 0.0)

        reranker_ms = ms_per_pair * pool_size
        total_lat_ms = reranker_ms + 4.07  # Hybrid first-stage (4.07ms) + Reranker

        primary_eval_results[f"top_{pool_size}"] = {
            "candidate_pool_size": pool_size,
            "metrics": macro_m,
            "zero_hits": {
                "zero_hit_at_10_count": zero_hits_10,
                "zero_hit_at_10_rate": (zero_hits_10 / num_queries) * 100,
                "zero_hit_at_100_count": zero_hits_100,
                "zero_hit_at_100_rate": (zero_hits_100 / num_queries) * 100,
            },
            "timing": {
                "reranker_inference_ms_per_query": reranker_ms,
                "first_stage_hybrid_ms": 4.07,
                "total_pipeline_latency_ms": total_lat_ms,
                "throughput_qps": 1000.0 / total_lat_ms,
            },
        }

        print(f"Top-{pool_size} Candidate Pool Results:")
        print(f"  nDCG@10:   {macro_m['nDCG@10']:.4f}")
        print(f"  MRR@10:    {macro_m['MRR@10']:.4f}")
        print(f"  Recall@5:  {macro_m['R@5']:.4f}  (Precision@5: {macro_m['P@5']:.4f})")
        print(f"  Recall@10: {macro_m['R@10']:.4f} (Precision@10: {macro_m['P@10']:.4f})")
        print(f"  Zero-hit@10: {zero_hits_10}/{num_queries} ({(zero_hits_10/num_queries)*100:.2f}%)")
        print(f"  Latency: {total_lat_ms:.2f} ms/query | QPS: {1000.0/total_lat_ms:.2f}")

        if pool_size == 20:
            predictions_top20 = preds

    # 6. Candidate Pool Scaling Benchmark: Top-10 vs Top-20 vs Top-50 on 100-query subset
    SCALING_QUERIES = 100
    print(f"\n--- Candidate Pool Scaling Benchmark: Top-10 vs Top-20 vs Top-50 on {SCALING_QUERIES} queries ---")
    
    # We need scores for candidates 21..50 for the first 100 queries
    scaling_extra_pairs = []
    scaling_meta = []
    for q_idx in range(SCALING_QUERIES):
        q_meta = query_meta_20[q_idx]
        q_text = q_meta["query_text"]
        top_100_ids = q_meta["top_100_ids"]
        cands_21_to_50 = top_100_ids[20:50]
        start_idx = len(scaling_extra_pairs)
        for art_id in cands_21_to_50:
            doc_text = corpus_dict.get(art_id, "")
            scaling_extra_pairs.append((q_text, doc_text))
        end_idx = len(scaling_extra_pairs)
        scaling_meta.append({
            "query_id": q_meta["query_id"],
            "start_idx": start_idx,
            "end_idx": end_idx,
        })

    cache_extra_path = PROJECT_ROOT / "results" / "benchmarks" / "reranker_extra_scores_cache.json"
    if cache_extra_path.exists():
        print(f"Loading cached extra scaling scores from {cache_extra_path}...")
        with open(cache_extra_path, "r", encoding="utf-8") as f:
            cache_extra_data = json.load(f)
        extra_scores = cache_extra_data["scores"]
        t_scale_dur = cache_extra_data["time_sec"]
    else:
        print(f"Scoring {len(scaling_extra_pairs)} additional pairs (ranks 21-50) for {SCALING_QUERIES} queries...")
        t_scale0 = time.time()
        extra_scores_raw = reranker.compute_score(
            [[q, d] for q, d in scaling_extra_pairs],
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
        )
        t_scale_dur = time.time() - t_scale0
        extra_scores = [float(s) for s in extra_scores_raw]
        atomic_write_json(cache_extra_path, {"scores": extra_scores, "time_sec": t_scale_dur})
        print(f"Saved extra scores cache to {cache_extra_path}")

    print(f"Additional pairs scored in {t_scale_dur:.2f}s ({len(scaling_extra_pairs)/max(t_scale_dur, 0.001):.1f} pairs/s).")

    # Evaluate Top-10, Top-20, Top-50 on these 100 queries
    scaling_comparison: Dict[str, Any] = {}
    for p_size in [10, 20, 50]:
        sub_metrics = []
        for q_idx in range(SCALING_QUERIES):
            q_meta = query_meta_20[q_idx]
            gt = q_meta["ground_truth"]
            top_100 = q_meta["top_100_ids"]
            start_i = q_meta["start_idx"]

            # First 20 scores from main run
            base_scores = scores_20[start_i : start_i + min(p_size, 20)]
            if p_size <= 20:
                sub_cand_ids = top_100[:p_size]
                sub_scores = base_scores[:p_size]
            else:
                # p_size == 50
                ext_meta = scaling_meta[q_idx]
                ext_s = extra_scores[ext_meta["start_idx"] : ext_meta["end_idx"]]
                sub_cand_ids = top_100[:50]
                sub_scores = base_scores + ext_s

            scored = sorted(zip(sub_cand_ids, sub_scores), key=lambda x: x[1], reverse=True)
            r_ids = [c[0] for c in scored] + top_100[p_size:]
            m = evaluate_query_retrieval(r_ids, gt)
            sub_metrics.append(m)

        sub_macro = aggregate_retrieval_metrics(sub_metrics)
        cand_lat = ms_per_pair * p_size + 4.07
        scaling_comparison[f"pool_{p_size}"] = {
            "candidate_pool_size": p_size,
            "queries": SCALING_QUERIES,
            "metrics": {
                "nDCG@10": sub_macro["nDCG@10"],
                "MRR@10": sub_macro["MRR@10"],
                "Recall@5": sub_macro["R@5"],
                "Recall@10": sub_macro["R@10"],
                "Precision@5": sub_macro["P@5"],
                "Precision@10": sub_macro["P@10"],
            },
            "latency_ms": cand_lat,
            "throughput_qps": 1000.0 / cand_lat,
        }
        print(f"Scaling Subset [Top-{p_size}]: nDCG@10={sub_macro['nDCG@10']:.4f} | MRR@10={sub_macro['MRR@10']:.4f} | R@5={sub_macro['R@5']:.4f} | Latency={cand_lat:.1f}ms")

    atomic_write_json(out_scaling_path, scaling_comparison)
    print(f"Saved scaling trade-off report to {out_scaling_path}")

    # 7. Win/Tie/Loss vs Baselines on all 438 queries
    print("\n--- Win/Tie/Loss Analysis on 438 Test Queries ---")
    win_loss_analysis = {}
    for pool_size in [10, 20]:
        pk = f"top_{pool_size}"
        w_ndcg, l_ndcg, t_ndcg = 0, 0, 0
        w_mrr, l_mrr, t_mrr = 0, 0, 0
        w_r5, l_r5, t_r5 = 0, 0, 0

        for i, q_meta in enumerate(query_meta_20):
            hyb_m = hybrid_predictions[i]["metrics"]

            subpool_ids = q_meta["top_100_ids"][:pool_size]
            start_i = q_meta["start_idx"]
            sub_scores = scores_20[start_i : start_i + pool_size]
            scored = sorted(zip(subpool_ids, sub_scores), key=lambda x: x[1], reverse=True)
            r_ids = [c[0] for c in scored] + q_meta["top_100_ids"][pool_size:]
            rerank_m = evaluate_query_retrieval(r_ids, q_meta["ground_truth"])

            # nDCG@10
            diff_ndcg = rerank_m["nDCG@10"] - hyb_m["nDCG@10"]
            if diff_ndcg > 1e-5:
                w_ndcg += 1
            elif diff_ndcg < -1e-5:
                l_ndcg += 1
            else:
                t_ndcg += 1

            # MRR@10
            diff_mrr = rerank_m["MRR@10"] - hyb_m["MRR@10"]
            if diff_mrr > 1e-5:
                w_mrr += 1
            elif diff_mrr < -1e-5:
                l_mrr += 1
            else:
                t_mrr += 1

            # R@5
            diff_r5 = rerank_m["R@5"] - hyb_m["R@5"]
            if diff_r5 > 1e-5:
                w_r5 += 1
            elif diff_r5 < -1e-5:
                l_r5 += 1
            else:
                t_r5 += 1

        win_loss_analysis[pk] = {
            "nDCG@10": {
                "wins": w_ndcg,
                "losses": l_ndcg,
                "ties": t_ndcg,
                "win_rate": (w_ndcg / num_queries) * 100,
                "loss_rate": (l_ndcg / num_queries) * 100,
            },
            "MRR@10": {
                "wins": w_mrr,
                "losses": l_mrr,
                "ties": t_mrr,
                "win_rate": (w_mrr / num_queries) * 100,
                "loss_rate": (l_mrr / num_queries) * 100,
            },
            "R@5": {
                "wins": w_r5,
                "losses": l_r5,
                "ties": t_r5,
                "win_rate": (w_r5 / num_queries) * 100,
                "loss_rate": (l_r5 / num_queries) * 100,
            },
        }
        print(f"Top-{pool_size} vs Hybrid: nDCG@10 Wins={w_ndcg} ({w_ndcg/num_queries*100:.1f}%), Losses={l_ndcg} ({l_ndcg/num_queries*100:.1f}%), Ties={t_ndcg}")
        print(f"Top-{pool_size} vs Hybrid: MRR@10  Wins={w_mrr} ({w_mrr/num_queries*100:.1f}%), Losses={l_mrr} ({l_mrr/num_queries*100:.1f}%), Ties={t_mrr}")
        print(f"Top-{pool_size} vs Hybrid: R@5     Wins={w_r5} ({w_r5/num_queries*100:.1f}%), Losses={l_r5} ({l_r5/num_queries*100:.1f}%), Ties={t_r5}")

    # 8. Load Baselines for Comprehensive Multi-System Comparison
    def load_metrics_file(path: Path) -> Dict[str, Any]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    dense_baseline = load_metrics_file(dense_metrics_path)
    bm25_baseline = load_metrics_file(bm25_metrics_path)
    hybrid_baseline = load_metrics_file(hybrid_metrics_path)

    bm25_m = bm25_baseline.get("metrics", {})
    dense_m = dense_baseline.get("metrics", {})
    hybrid_m = hybrid_baseline.get("metrics", {})

    downstream_rag_impact = {
        "description": "Downstream context impact when feeding Top-5 articles to LLM context window",
        "systems": {
            "bm25_tuned": {
                "context_precision_at_5": bm25_m.get("P@5", 0.0),
                "context_recall_at_5": bm25_m.get("R@5", 0.0),
            },
            "dense_bge_m3": {
                "context_precision_at_5": dense_m.get("P@5", 0.0),
                "context_recall_at_5": dense_m.get("R@5", 0.0),
            },
            "hybrid_weighted_rrf": {
                "context_precision_at_5": hybrid_m.get("P@5", 0.0),
                "context_recall_at_5": hybrid_m.get("R@5", 0.0),
            },
            "hybrid_plus_reranker_top10": {
                "context_precision_at_5": primary_eval_results["top_10"]["metrics"]["P@5"],
                "context_recall_at_5": primary_eval_results["top_10"]["metrics"]["R@5"],
            },
            "hybrid_plus_reranker_top20": {
                "context_precision_at_5": primary_eval_results["top_20"]["metrics"]["P@5"],
                "context_recall_at_5": primary_eval_results["top_20"]["metrics"]["R@5"],
            },
        },
    }

    # 9. Save Official Artifacts
    # Primary production config: Top-20 candidate pool into reranker
    chosen_config = primary_eval_results["top_20"]
    official_metrics = {
        "model": "Hybrid Retriever + Cross-Encoder Reranker (BAAI/bge-reranker-base)",
        "configuration": {
            "first_stage": "Hybrid Weighted RRF (k=5, w_dense=0.75, w_bm25=0.25)",
            "reranker_model": "BAAI/bge-reranker-base",
            "candidate_pool_size": 20,
            "max_seq_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "device": device,
            "fp16": torch.cuda.is_available(),
        },
        "benchmark": {
            "name": "retrieval_test.json",
            "num_queries": num_queries,
        },
        "timing": chosen_config["timing"],
        "metrics": chosen_config["metrics"],
        "zero_hits": chosen_config["zero_hits"],
        "candidate_pool_comparison": primary_eval_results,
    }

    atomic_write_json(out_metrics_path, official_metrics)
    print(f"\nSaved official metrics to {out_metrics_path}")

    atomic_write_json(out_preds_path, predictions_top20)
    print(f"Saved predictions to {out_preds_path}")

    comp_data = {
        "evaluation_dataset": "retrieval_test.json (438 queries)",
        "models": {
            "BM25_Tuned": {
                "nDCG@10": bm25_m.get("nDCG@10"),
                "MRR@10": bm25_m.get("MRR@10"),
                "Recall@5": bm25_m.get("R@5"),
                "Recall@10": bm25_m.get("R@10"),
                "Recall@20": bm25_m.get("R@20"),
                "Recall@100": bm25_m.get("R@100"),
                "Precision@5": bm25_m.get("P@5"),
                "Zero_hit@10_rate": 31.96,
                "Latency_ms": 3.26,
            },
            "Dense_BGE_M3": {
                "nDCG@10": dense_m.get("nDCG@10"),
                "MRR@10": dense_m.get("MRR@10"),
                "Recall@5": dense_m.get("R@5"),
                "Recall@10": dense_m.get("R@10"),
                "Recall@20": dense_m.get("R@20"),
                "Recall@100": dense_m.get("R@100"),
                "Precision@5": dense_m.get("P@5"),
                "Zero_hit@10_rate": 13.93,
                "Latency_ms": 17.50,
            },
            "Hybrid_Weighted_RRF": {
                "nDCG@10": hybrid_m.get("nDCG@10"),
                "MRR@10": hybrid_m.get("MRR@10"),
                "Recall@5": hybrid_m.get("R@5"),
                "Recall@10": hybrid_m.get("R@10"),
                "Recall@20": hybrid_m.get("R@20"),
                "Recall@100": hybrid_m.get("R@100"),
                "Precision@5": hybrid_m.get("P@5"),
                "Zero_hit@10_rate": 12.33,
                "Latency_ms": 4.07,
            },
            "Hybrid_Plus_Reranker_Top10": {
                "nDCG@10": primary_eval_results["top_10"]["metrics"]["nDCG@10"],
                "MRR@10": primary_eval_results["top_10"]["metrics"]["MRR@10"],
                "Recall@5": primary_eval_results["top_10"]["metrics"]["R@5"],
                "Recall@10": primary_eval_results["top_10"]["metrics"]["R@10"],
                "Recall@20": primary_eval_results["top_10"]["metrics"]["R@20"],
                "Recall@100": primary_eval_results["top_10"]["metrics"]["R@100"],
                "Precision@5": primary_eval_results["top_10"]["metrics"]["P@5"],
                "Zero_hit@10_rate": primary_eval_results["top_10"]["zero_hits"]["zero_hit_at_10_rate"],
                "Latency_ms": primary_eval_results["top_10"]["timing"]["total_pipeline_latency_ms"],
            },
            "Hybrid_Plus_Reranker_Top20": {
                "nDCG@10": primary_eval_results["top_20"]["metrics"]["nDCG@10"],
                "MRR@10": primary_eval_results["top_20"]["metrics"]["MRR@10"],
                "Recall@5": primary_eval_results["top_20"]["metrics"]["R@5"],
                "Recall@10": primary_eval_results["top_20"]["metrics"]["R@10"],
                "Recall@20": primary_eval_results["top_20"]["metrics"]["R@20"],
                "Recall@100": primary_eval_results["top_20"]["metrics"]["R@100"],
                "Precision@5": primary_eval_results["top_20"]["metrics"]["P@5"],
                "Zero_hit@10_rate": primary_eval_results["top_20"]["zero_hits"]["zero_hit_at_10_rate"],
                "Latency_ms": primary_eval_results["top_20"]["timing"]["total_pipeline_latency_ms"],
            },
        },
        "win_loss_vs_hybrid": win_loss_analysis,
        "downstream_rag_impact": downstream_rag_impact,
    }

    atomic_write_json(out_comp_path, comp_data)
    print(f"Saved comparative report to {out_comp_path}")
    print("\n" + "=" * 80)
    print("PHASE 12 EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_reranker_evaluation()
