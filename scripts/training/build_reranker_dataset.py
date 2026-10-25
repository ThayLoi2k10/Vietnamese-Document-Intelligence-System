"""Xây dựng tập dữ liệu huấn luyện & validation cho Cross-Encoder Reranker (In-domain Hard Negatives).

Chạy hoàn toàn ở Local:
- Nguồn:
  * `data/benchmarks/retrieval_dev_train.json` (1,401 câu train)
  * `data/benchmarks/retrieval_dev_val.json` (351 câu validation)
  * `data/benchmarks/retrieval_test.json` (438 câu test - chỉ dùng để kiểm tra Zero Data Leakage)
  * `data/indices/bm25_index.pkl` (BM25 tuned k1=1.80, b=0.85)
  * `results/benchmarks/dense_train_predictions.json` (BGE-M3 Top-100 trên 1,752 câu train)
  * `data/processed/legal_retrieval/corpus_articles_preprocessed.parquet` (59,636 điều luật)
- Đầu ra tại `data/training/`:
  * `reranker_train_groups.jsonl`: 1,401 nhóm (query, positives, hard_negatives từ Top-20 Hybrid RRF)
  * `reranker_val_candidates.jsonl`: 351 câu val kèm Top-20 ứng viên Hybrid RRF để chấm nDCG@10/Hit@3 mỗi epoch
  * `dataset_manifest.json`: thống kê & kiểm chứng Zero Leakage so với `retrieval_test.json`
"""

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import pyarrow.parquet as pq

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    evaluate_query_retrieval,
)
from src.retrieval.bm25_retriever import BM25Retriever

K_RRF = 5
W_DENSE = 0.75
W_BM25 = 0.25
CANDIDATE_DEPTH = 100
POOL_TOP_K = 20
MAX_HARD_NEGATIVES = 7
MAX_CHAR_LENGTH = 1600  # ~384-512 tokens tiếng Việt


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def fuse_weighted_rrf(
    dense_ids: List[int], bm25_ids: List[int], top_k: int = POOL_TOP_K
) -> List[tuple[int, float]]:
    scores: Dict[int, float] = defaultdict(float)
    for rank, aid in enumerate(dense_ids[:CANDIDATE_DEPTH], start=1):
        scores[int(aid)] += W_DENSE / (K_RRF + rank)
    for rank, aid in enumerate(bm25_ids[:CANDIDATE_DEPTH], start=1):
        scores[int(aid)] += W_BM25 / (K_RRF + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def truncate_legal_text(text: str, max_chars: int = MAX_CHAR_LENGTH) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    # Giữ phần đầu (tiêu đề Điều + các khoản đầu) và phần cuối điều luật
    head = int(max_chars * 0.75)
    tail = max_chars - head
    return text[:head].rstrip() + "\n...\n" + text[-tail:].lstrip()


def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("BƯỚC 1 (LOCAL): KHAI PHÁ HARD NEGATIVES & XÂY DỰNG TẬP HUẤN LUYỆN RERANKER")
    print("=" * 80)

    train_split_path = PROJECT_ROOT / "data" / "benchmarks" / "retrieval_dev_train.json"
    val_split_path = PROJECT_ROOT / "data" / "benchmarks" / "retrieval_dev_val.json"
    test_split_path = PROJECT_ROOT / "data" / "benchmarks" / "retrieval_test.json"
    dense_preds_path = PROJECT_ROOT / "results" / "benchmarks" / "dense_train_predictions.json"
    bm25_path = PROJECT_ROOT / "data" / "indices" / "bm25_index.pkl"
    corpus_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "legal_retrieval"
        / "corpus_articles_preprocessed.parquet"
    )
    out_dir = PROJECT_ROOT / "data" / "training"
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_train = json.loads(train_split_path.read_text(encoding="utf-8"))
    dev_val = json.loads(val_split_path.read_text(encoding="utf-8"))
    test_set = json.loads(test_split_path.read_text(encoding="utf-8"))

    train_qids = {int(q["query_id"]) for q in dev_train}
    val_qids = {int(q["query_id"]) for q in dev_val}
    test_qids = {int(q["query_id"]) for q in test_set}

    if train_qids & val_qids:
        raise RuntimeError("Phát hiện trùng lặp query_id giữa dev_train và dev_val!")
    if (train_qids | val_qids) & test_qids:
        raise RuntimeError("Phát hiện rò rỉ dữ liệu (Data Leakage) với retrieval_test.json!")
    print(
        f"[OK] Kiểm tra chống rò rỉ (Zero Leakage): "
        f"dev_train={len(train_qids)}, dev_val={len(val_qids)}, frozen_test={len(test_qids)} (Overlap = 0)"
    )

    print("Đang nạp corpus 59,636 điều luật...")
    table = pq.read_table(
        corpus_path, columns=["global_article_id", "law_id", "aid", "text"]
    )
    corpus_text: Dict[int, str] = {}
    for gid, lid, aid, txt in zip(
        table.column("global_article_id").to_pylist(),
        table.column("law_id").to_pylist(),
        table.column("aid").to_pylist(),
        table.column("text").to_pylist(),
    ):
        header = f"[Văn bản: {lid} | Điều: {aid}] " if lid else ""
        corpus_text[int(gid)] = header + str(txt or "")

    print("Đang nạp dự đoán Dense BGE-M3 trên tập train...")
    dense_raw = json.loads(dense_preds_path.read_text(encoding="utf-8"))
    dense_by_qid: Dict[int, List[int]] = {
        int(item["query_id"]): [int(x) for x in item["top_100_retrieved_ids"]]
        for item in dense_raw
    }

    print("Đang nạp chỉ mục BM25 Tuned (k1=1.80, b=0.85) và chạy truy xuất Top-100...")
    bm25 = BM25Retriever.load(bm25_path)
    bm25.k1 = 1.80
    bm25.b = 0.85

    # Xây dựng tập Train Groups (1,401 câu)
    train_groups: List[Dict[str, Any]] = []
    total_pos = 0
    total_hard_neg = 0

    for q in dev_train:
        qid = int(q["query_id"])
        qtext = str(q["query_text"])
        rel_ids = [int(x) for x in q["relevant_article_ids"]]
        rel_set = set(rel_ids)

        bm25_res = bm25.search(qtext, top_k=CANDIDATE_DEPTH)
        bm25_ids = [r.article_id for r in bm25_res]
        dense_ids = dense_by_qid.get(qid, [])

        fused = fuse_weighted_rrf(dense_ids, bm25_ids, top_k=POOL_TOP_K)
        fused_score_map = dict(fused)

        positives = []
        for rank_idx, (aid, sc) in enumerate(fused, start=1):
            if aid in rel_set:
                positives.append(
                    {
                        "article_id": aid,
                        "hybrid_rank": rank_idx,
                        "hybrid_score": round(float(sc), 6),
                        "text": truncate_legal_text(corpus_text.get(aid, "")),
                    }
                )
        # Bổ sung các ground-truth không lọt vào Top-20 để model luôn học được đủ đáp án đúng
        in_pool_pos = {p["article_id"] for p in positives}
        for aid in rel_ids:
            if aid not in in_pool_pos and aid in corpus_text:
                positives.append(
                    {
                        "article_id": aid,
                        "hybrid_rank": POOL_TOP_K + 5,
                        "hybrid_score": 0.01,
                        "text": truncate_legal_text(corpus_text[aid]),
                    }
                )

        hard_negatives = []
        for rank_idx, (aid, sc) in enumerate(fused, start=1):
            if aid not in rel_set:
                hard_negatives.append(
                    {
                        "article_id": aid,
                        "hybrid_rank": rank_idx,
                        "hybrid_score": round(float(sc), 6),
                        "text": truncate_legal_text(corpus_text.get(aid, "")),
                    }
                )
                if len(hard_negatives) >= MAX_HARD_NEGATIVES:
                    break

        if positives and hard_negatives:
            train_groups.append(
                {
                    "query_id": qid,
                    "query_text": qtext,
                    "positives": positives,
                    "hard_negatives": hard_negatives,
                }
            )
            total_pos += len(positives)
            total_hard_neg += len(hard_negatives)

    # Xây dựng tập Validation Candidates (351 câu) + đo baseline Hybrid RRF trên Val
    val_records: List[Dict[str, Any]] = []
    val_per_query_metrics = []
    val_hit1 = val_hit3 = 0

    for q in dev_val:
        qid = int(q["query_id"])
        qtext = str(q["query_text"])
        rel_ids = [int(x) for x in q["relevant_article_ids"]]
        rel_set = set(rel_ids)

        bm25_res = bm25.search(qtext, top_k=CANDIDATE_DEPTH)
        bm25_ids = [r.article_id for r in bm25_res]
        dense_ids = dense_by_qid.get(qid, [])

        fused = fuse_weighted_rrf(dense_ids, bm25_ids, top_k=POOL_TOP_K)
        cand_ids = [aid for aid, _ in fused]

        val_hit1 += int(bool(set(cand_ids[:1]) & rel_set))
        val_hit3 += int(bool(set(cand_ids[:3]) & rel_set))
        val_per_query_metrics.append(
            evaluate_query_retrieval(
                retrieved_ids=cand_ids,
                relevant_ids=rel_set,
                k_list=[1, 5, 10, 20],
            )
        )

        candidates = [
            {
                "article_id": aid,
                "hybrid_rank": rank_idx,
                "hybrid_score": round(float(sc), 6),
                "label": 1 if aid in rel_set else 0,
                "text": truncate_legal_text(corpus_text.get(aid, "")),
            }
            for rank_idx, (aid, sc) in enumerate(fused, start=1)
        ]
        val_records.append(
            {
                "query_id": qid,
                "query_text": qtext,
                "relevant_article_ids": rel_ids,
                "candidates": candidates,
            }
        )

    val_agg = aggregate_retrieval_metrics(val_per_query_metrics)
    n_val = len(val_records)

    train_out = out_dir / "reranker_train_groups.jsonl"
    val_out = out_dir / "reranker_val_candidates.jsonl"
    manifest_out = out_dir / "dataset_manifest.json"

    with open(train_out, "w", encoding="utf-8") as f:
        for row in train_groups:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(val_out, "w", encoding="utf-8") as f:
        for row in val_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_splits": {
            "dev_train_queries": len(dev_train),
            "dev_val_queries": len(dev_val),
            "frozen_test_queries_excluded": len(test_set),
            "zero_test_leakage_verified": True,
        },
        "hybrid_teacher_config": {
            "k_rrf": K_RRF,
            "w_dense": W_DENSE,
            "w_bm25": W_BM25,
            "candidate_pool_size": POOL_TOP_K,
            "max_hard_negatives_per_query": MAX_HARD_NEGATIVES,
        },
        "train_stats": {
            "num_query_groups": len(train_groups),
            "total_positives": total_pos,
            "total_hard_negatives": total_hard_neg,
            "total_Query_Article_pairs": total_pos + total_hard_neg,
            "file_size_bytes": train_out.stat().st_size,
            "sha256": sha256_file(train_out),
        },
        "val_baseline_hybrid_rrf": {
            "num_queries": n_val,
            "Hit@1_pct": round(val_hit1 / n_val * 100, 2),
            "Hit@3_pct": round(val_hit3 / n_val * 100, 2),
            "nDCG@10": round(float(val_agg["nDCG@10"]), 4),
            "MRR@10": round(float(val_agg["MRR@10"]), 4),
            "R@10": round(float(val_agg["R@10"]), 4),
            "file_size_bytes": val_out.stat().st_size,
            "sha256": sha256_file(val_out),
        },
    }
    manifest_out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    elapsed = time.time() - t0
    print("-" * 80)
    print(f"Hoàn tất trong {elapsed:.1f}s!")
    print(
        f"1. Tập Train: {train_out.name} ({train_out.stat().st_size / 1024 / 1024:.2f} MB) | "
        f"{len(train_groups)} queries | {total_pos} positives | {total_hard_neg} hard negatives"
    )
    print(
        f"2. Tập Val  : {val_out.name} ({val_out.stat().st_size / 1024 / 1024:.2f} MB) | "
        f"{n_val} queries x Top-{POOL_TOP_K} candidates"
    )
    print(
        f"3. Baseline Hybrid RRF trên Val (351 câu): "
        f"Hit@1 = {manifest['val_baseline_hybrid_rrf']['Hit@1_pct']}% | "
        f"Hit@3 = {manifest['val_baseline_hybrid_rrf']['Hit@3_pct']}% | "
        f"nDCG@10 = {manifest['val_baseline_hybrid_rrf']['nDCG@10']}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
