"""Run grid search tuning for hybrid retrieval fusion weights on training set."""

from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

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

from src.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    evaluate_query_retrieval,
)
from src.retrieval.bm25_retriever import BM25Retriever


def atomic_write_json(file_path: Path, data: Any) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=file_path.parent, delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp_name = tmp.name
    os.replace(tmp_name, file_path)


def run_tuning() -> None:
    train_path = PROJECT_ROOT / "data" / "benchmarks" / "retrieval_train.json"
    dense_preds_path = PROJECT_ROOT / "results" / "benchmarks" / "dense_train_predictions.json"
    bm25_index_path = PROJECT_ROOT / "data" / "indices" / "bm25_index.pkl"
    output_path = PROJECT_ROOT / "results" / "benchmarks" / "hybrid_tuning_train_results.json"

    print("=" * 70)
    print("BẮT ĐẦU TINH CHỈNH SIÊU THAM SỐ HYBRID RETRIEVAL TRÊN TẬP TRAIN (1,752 queries)")
    print("=" * 70)

    # 1. Nạp tập train
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    print(f"[1/4] Đã nạp {len(train_data)} câu truy vấn train.")

    # 2. Nạp kết quả Dense train đã lưu
    with open(dense_preds_path, "r", encoding="utf-8") as f:
        dense_train = json.load(f)
    dense_dict = {q["query_id"]: q["top_100_retrieved_ids"] for q in dense_train}
    print(f"[2/4] Đã nạp kết quả Dense top-100 cho {len(dense_dict)} câu truy vấn.")

    # 3. Chạy BM25 top-100 cho toàn bộ câu train
    print("[3/4] Đang thực hiện tìm kiếm BM25 tuned (k1=1.80, b=0.85)...")
    queries = [q["query_text"] for q in train_data]
    bm25 = BM25Retriever.load(bm25_index_path)
    bm25.k1 = 1.80
    bm25.b = 0.85
    t0 = time.time()
    bm25_results = bm25.batch_search(queries, top_k=100)
    bm25_duration = time.time() - t0
    bm25_dict = {
        q["query_id"]: [r.article_id for r in res]
        for q, res in zip(train_data, bm25_results)
    }
    print(f"      Hoàn thành BM25 batch search trong {bm25_duration:.2f}s ({bm25_duration/len(queries)*1000:.2f} ms/query).")

    # 4. Grid search RRF k và trọng số w_dense : w_bm25
    print("[4/4] Bắt đầu Grid Search trên tập Train...")
    k_candidates = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
    weight_candidates = [
        (0.50, 0.50),  # Standard RRF
        (0.60, 0.40),
        (0.70, 0.30),
        (0.75, 0.25),
        (0.80, 0.20),
        (0.85, 0.15),
        (0.90, 0.10),
    ]

    all_grid_results = []
    best_config = None
    best_train_ndcg = -1.0

    total_runs = len(k_candidates) * len(weight_candidates)
    run_idx = 0

    for k_val in k_candidates:
        for wd, wb in weight_candidates:
            run_idx += 1
            per_query_metrics = []
            for q in train_data:
                qid = q["query_id"]
                dense_ids = dense_dict[qid]
                bm25_ids = bm25_dict[qid]

                scores = defaultdict(float)
                for rank, aid in enumerate(dense_ids, start=1):
                    scores[aid] += wd / (k_val + rank)
                for rank, aid in enumerate(bm25_ids, start=1):
                    scores[aid] += wb / (k_val + rank)

                sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]
                fused_ids = [aid for aid, sc in sorted_items]
                m = evaluate_query_retrieval(fused_ids, q["relevant_article_ids"])
                per_query_metrics.append(m)

            agg = aggregate_retrieval_metrics(per_query_metrics)
            entry = {
                "fusion_method": "rrf" if wd == 0.5 and wb == 0.5 else "weighted_rrf",
                "k_rrf": k_val,
                "w_dense": wd,
                "w_bm25": wb,
                "metrics": {
                    "nDCG@10": agg["nDCG@10"],
                    "R@10": agg["R@10"],
                    "MRR@10": agg["MRR@10"],
                    "R@100": agg["R@100"],
                    "MAP@10": agg["MAP@10"],
                    "nDCG@100": agg["nDCG@100"],
                },
            }
            all_grid_results.append(entry)

            if agg["nDCG@10"] > best_train_ndcg:
                best_train_ndcg = agg["nDCG@10"]
                best_config = entry
                print(
                    f"  [{run_idx}/{total_runs}] NEW BEST: k={k_val:2d}, w_dense={wd:.2f}, w_bm25={wb:.2f} "
                    f"-> Train nDCG@10={agg['nDCG@10']:.4f}, R@10={agg['R@10']:.4f}, MRR@10={agg['MRR@10']:.4f}, R@100={agg['R@100']:.4f}"
                )

    tuning_report = {
        "benchmark": "retrieval_train.json",
        "num_queries": len(train_data),
        "selection_metric": "nDCG@10",
        "best_configuration": best_config,
        "grid_results": all_grid_results,
    }

    atomic_write_json(output_path, tuning_report)
    print("\n" + "=" * 70)
    print("HOÀN THÀNH TINH CHỈNH TRÊN TẬP TRAIN!")
    print(f"Cấu hình tối ưu được chọn: k={best_config['k_rrf']}, w_dense={best_config['w_dense']:.2f}, w_bm25={best_config['w_bm25']:.2f}")
    print(f"Metrics trên Train: nDCG@10={best_config['metrics']['nDCG@10']:.4f}, R@10={best_config['metrics']['R@10']:.4f}, MRR@10={best_config['metrics']['MRR@10']:.4f}, R@100={best_config['metrics']['R@100']:.4f}")
    print(f"Đã lưu báo cáo tinh chỉnh vào: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_tuning()
