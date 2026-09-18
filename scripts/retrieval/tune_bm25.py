# scripts/retrieval/tune_bm25.py
"""Script Grid Search tinh chỉnh siêu tham số BM25 (k1, b) trên tập Train.

Thực hiện:
    1. Nạp training benchmark (retrieval_train.json) và BM25 index đã build
    2. Tiền xử lý và tạo cache postings cho toàn bộ query để tăng tốc tối đa
    3. Quét lưới siêu tham số (k1 x b) và tính toán vector hóa điểm số BM25
    4. Chỉ xếp hạng các tài liệu có score > 0 (tránh tính điểm ảo cho zero-score docs)
    5. Đánh giá chất lượng bằng nDCG@10, Recall@10 và MAP@10
    6. Chọn cấu hình tối ưu, so sánh đối chứng với default config (k1=1.5, b=0.75)
    7. Ghi an toàn kết quả qua file tạm và atomic replace
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Set
import numpy as np

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

# Thêm PROJECT_ROOT vào sys.path để import được src
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import (
    compute_ap_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
)
from src.retrieval.bm25_retriever import BM25Retriever, default_tokenizer


DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "benchmarks" / "retrieval_train.json"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "indices" / "bm25_index.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmarks"
DEFAULT_OUTPUT_REPORT = DEFAULT_OUTPUT_DIR / "bm25_tuning_train_results.json"

DEFAULT_K1_GRID = [0.6, 0.9, 1.2, 1.5, 1.8, 2.1]
DEFAULT_B_GRID = [0.40, 0.55, 0.70, 0.75, 0.85]


def evaluate_candidate(
    cached_queries: List[Dict[str, Any]],
    k1: float,
    b: float,
    avgdl: float,
    corpus_size: int,
    article_ids: np.ndarray,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Tính điểm và đo metric trung bình trên tập cached queries cho 1 cặp (k1, b)."""
    ndcg_list: List[float] = []
    recall_list: List[float] = []
    map_list: List[float] = []

    for item in cached_queries:
        postings = item["postings"]
        ground_truth: Set[int] = item["ground_truth"]

        if not postings:
            ndcg_list.append(0.0)
            recall_list.append(0.0)
            map_list.append(0.0)
            continue

        scores = np.zeros(corpus_size, dtype=np.float32)

        for p in postings:
            docs = p["docs"]
            tfs = p["tfs"]
            idf = p["idf"]
            lens = p["lens"]

            # Công thức BM25 vector hóa
            denom = tfs + k1 * (1.0 - b + b * (lens / avgdl))
            scores[docs] += idf * ((tfs * (k1 + 1.0)) / denom)

        # Lọc chỉ lấy các tài liệu có điểm thực sự dương
        positive_mask = scores > 0.0
        positive_indices = np.where(positive_mask)[0]

        if len(positive_indices) == 0:
            ndcg_list.append(0.0)
            recall_list.append(0.0)
            map_list.append(0.0)
            continue

        positive_scores = scores[positive_indices]
        k_eval = min(top_k, len(positive_indices))

        if len(positive_indices) > k_eval:
            # Dùng argpartition trên tập dương để tăng tốc
            partial_idx = np.argpartition(positive_scores, -k_eval)[-k_eval:]
            sorted_within_top = partial_idx[np.argsort(-positive_scores[partial_idx])]
            final_doc_indices = positive_indices[sorted_within_top]
        else:
            sorted_order = np.argsort(-positive_scores)
            final_doc_indices = positive_indices[sorted_order]

        retrieved_ids = article_ids[final_doc_indices].tolist()

        ndcg_list.append(compute_ndcg_at_k(retrieved_ids, ground_truth, k=top_k))
        recall_list.append(compute_recall_at_k(retrieved_ids, ground_truth, k=top_k))
        map_list.append(compute_ap_at_k(retrieved_ids, ground_truth, k=top_k))

    n = len(cached_queries)
    return {
        "k1": round(k1, 2),
        "b": round(b, 2),
        f"nDCG@{top_k}": round(sum(ndcg_list) / n, 4),
        f"Recall@{top_k}": round(sum(recall_list) / n, 4),
        f"MAP@{top_k}": round(sum(map_list) / n, 4),
    }


def tune_bm25(
    train_path: Path = DEFAULT_TRAIN_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
    output_report: Path = DEFAULT_OUTPUT_REPORT,
    k1_grid: List[float] = DEFAULT_K1_GRID,
    b_grid: List[float] = DEFAULT_B_GRID,
    top_k: int = 10,
) -> int:
    """Thực thi Grid Search tinh chỉnh tham số BM25."""
    print("=" * 70)
    print("BƯỚC 5.9: GRID SEARCH TINH CHỈNH SIÊU THAM SỐ BM25 (TRÊN TRAIN SET)")
    print("=" * 70)

    output_report.parent.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        print(f"[FAIL] Không tìm thấy training benchmark: {train_path}")
        return 1
    if not index_path.exists():
        print(f"[FAIL] Không tìm thấy BM25 index: {index_path}")
        return 1

    # 1. Nạp tập Train và Retriever
    print(f"1. Nạp training benchmark: {train_path.name}...")
    try:
        with open(train_path, "r", encoding="utf-8") as f:
            train_queries = json.load(f)
    except Exception as e:
        print(f"[FAIL] Lỗi đọc train benchmark: {e}")
        return 1

    print(f"   -> Số lượng queries huấn luyện: {len(train_queries):,}")
    if len(train_queries) == 0:
        print("[FAIL] Tập train queries rỗng.")
        return 1

    print(f"2. Nạp BM25 index từ: {index_path.name}...")
    try:
        retriever = BM25Retriever.load(index_path)
    except Exception as e:
        print(f"[FAIL] Lỗi nạp BM25 index: {e}")
        return 1

    avgdl = retriever.avgdl
    corpus_size = retriever.corpus_size
    article_ids = np.array(retriever.article_ids)
    print(f"   -> Corpus size: {corpus_size:,} docs | avgdl: {avgdl:.2f} words")

    # 2. Xây dựng Query Postings Cache để tăng tốc
    print(f"\n3. Tiền xử lý và tạo cache postings cho {len(train_queries):,} queries...")
    cache_start = time.time()
    cached_queries = []

    for item in train_queries:
        tokens = default_tokenizer(item.get("query_text", ""))
        postings = []
        for term in set(tokens):
            if term in retriever.inverted_index:
                p = retriever.inverted_index[term]
                postings.append(
                    {
                        "docs": p["docs"],
                        "tfs": p["tfs"],
                        "idf": retriever.idf[term],
                        "lens": retriever.doc_lengths[p["docs"]],
                    }
                )

        cached_queries.append(
            {
                "ground_truth": set(item.get("relevant_article_ids", [])),
                "postings": postings,
            }
        )

    print(f"   -> Tạo cache hoàn tất trong {time.time() - cache_start:.2f}s")

    # 3. Quét lưới siêu tham số (Grid Search)
    total_configs = len(k1_grid) * len(b_grid)
    print(f"\n4. Bắt đầu Grid Search {total_configs} cấu hình (k1 x b)...")
    grid_start = time.time()
    results = []

    for k1 in k1_grid:
        for b in b_grid:
            config_res = evaluate_candidate(
                cached_queries=cached_queries,
                k1=k1,
                b=b,
                avgdl=avgdl,
                corpus_size=corpus_size,
                article_ids=article_ids,
                top_k=top_k,
            )
            results.append(config_res)

    total_tuning_time = time.time() - grid_start
    print(
        f"   -> Hoàn thành {total_configs} cấu hình trong {total_tuning_time:.2f}s "
        f"({total_tuning_time / total_configs:.2f}s / config)"
    )

    # 4. Sắp xếp và chọn cấu hình tối ưu theo nDCG@k
    metric_key = f"nDCG@{top_k}"
    results.sort(key=lambda x: x[metric_key], reverse=True)
    best_config = results[0]

    # Tìm lại kết quả của cấu hình mặc định (k1=1.5, b=0.75) để so sánh
    default_train_config = next(
        (r for r in results if r["k1"] == 1.5 and r["b"] == 0.75), None
    )

    # 5. Lưu kết quả ra file qua temporary replacement
    output_payload = {
        "dataset": train_path.name,
        "num_queries": len(train_queries),
        "total_configs": total_configs,
        "tuning_time_sec": round(total_tuning_time, 2),
        "best_config": best_config,
        "default_config": default_train_config,
        "all_results": results,
    }

    tmp_output = output_report.with_suffix(".json.tmp")
    try:
        with open(tmp_output, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)
        tmp_output.replace(output_report)
    finally:
        if tmp_output.exists():
            tmp_output.unlink(missing_ok=True)

    # 6. In bảng kết quả Top 10 cấu hình tốt nhất
    print("\n" + "=" * 70)
    print("KẾT QUẢ TOP 10 CẤU HÌNH SIÊU THAM SỐ TRÊN TẬP TRAIN")
    print("=" * 70)
    recall_key = f"Recall@{top_k}"
    map_key = f"MAP@{top_k}"
    print(
        f"{'Rank':<5} | {'k1':<6} | {'b':<6} | {metric_key:<10} | {recall_key:<10} | {map_key:<10}"
    )
    print("-" * 70)
    for idx, r in enumerate(results[:10], start=1):
        is_best = " (BEST)" if idx == 1 else ""
        print(
            f"{idx:<5} | {r['k1']:<6.2f} | {r['b']:<6.2f} | "
            f"{r[metric_key]:<10.4f} | {r[recall_key]:<10.4f} | {r[map_key]:<10.4f}{is_best}"
        )

    print("-" * 70)
    if default_train_config:
        delta = best_config[metric_key] - default_train_config[metric_key]
        print(f"Cấu hình mặc định (Default: k1=1.5, b=0.75) trên Train:")
        print(
            f"  -> {metric_key}: {default_train_config[metric_key]:.4f} | "
            f"{recall_key}: {default_train_config[recall_key]:.4f} | "
            f"{map_key}: {default_train_config[map_key]:.4f}"
        )
        print(f"\nCấu hình tối ưu được chọn:")
        print(f"  -> k1* = {best_config['k1']} | b* = {best_config['b']}")
        print(f"  -> {metric_key} Train: {best_config[metric_key]:.4f} (Chênh lệch: {delta:+.4f})")

    print("=" * 70)
    print(f"Đã lưu kết quả chi tiết: {output_report}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid Search tinh chỉnh siêu tham số BM25 trên tập Train."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="Đường dẫn file retrieval_train.json.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Đường dẫn file bm25_index.pkl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_REPORT,
        help="Đường dẫn lưu kết quả JSON.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Ngưỡng đánh giá xếp hạng k (mặc định: 10).",
    )
    args = parser.parse_args()

    exit_code = tune_bm25(
        train_path=args.train,
        index_path=args.index,
        output_report=args.output,
        top_k=args.top_k,
    )
    sys.exit(exit_code)