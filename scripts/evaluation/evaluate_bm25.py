# scripts/evaluation/evaluate_bm25.py
"""Script đánh giá thực nghiệm mô hình BM25 Retriever trên Hold-out Test Set.

Thực hiện:
    1. Nạp hold-out test benchmark (retrieval_test.json) và BM25 index
    2. Guard chống tập test rỗng (ZeroDivisionError protection)
    3. Truy vấn Top-K cho từng test query, đo latency chính xác bằng time.perf_counter()
    4. Tính toán hệ thống metrics IR chuẩn (P@k, R@k, MRR@k, MAP@k, nDCG@k)
    5. Đối chiếu kết quả trực tiếp với TF-IDF baseline (Head-to-head comparison)
    6. Xuất báo cáo tổng hợp và log chi tiết qua cơ chế ghi nguyên tử (atomic replace)
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

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
    aggregate_retrieval_metrics,
    evaluate_query_retrieval,
)
from src.retrieval.bm25_retriever import BM25Retriever


DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "benchmarks" / "retrieval_test.json"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "indices" / "bm25_index.pkl"
DEFAULT_TFIDF_METRICS_PATH = (
    PROJECT_ROOT / "results" / "benchmarks" / "tfidf_test_metrics.json"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmarks"
DEFAULT_METRICS_OUTPUT = DEFAULT_OUTPUT_DIR / "bm25_default_test_metrics.json"
DEFAULT_PREDICTIONS_OUTPUT = DEFAULT_OUTPUT_DIR / "bm25_default_test_predictions.json"

DEFAULT_TOP_K = 100
DEFAULT_EVAL_K_LIST = [1, 5, 10, 20, 50, 100]

METRICS_TO_COMPARE = [
    "P@1", "P@5", "P@10",
    "R@1", "R@5", "R@10", "R@20", "R@50", "R@100",
    "MRR@10", "MAP@10", "MAP@100", "nDCG@10", "nDCG@100"
]


def evaluate_bm25(
    test_path: Path = DEFAULT_TEST_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
    tfidf_metrics_path: Path = DEFAULT_TFIDF_METRICS_PATH,
    metrics_output: Path = DEFAULT_METRICS_OUTPUT,
    predictions_output: Path = DEFAULT_PREDICTIONS_OUTPUT,
    top_k: int = DEFAULT_TOP_K,
    eval_k_list: Optional[List[int]] = None,
    model_name: str = "BM25 (Default: k1=1.5, b=0.75)",
) -> int:
    """Thực thi đánh giá thực nghiệm BM25 trên hold-out test set."""
    if eval_k_list is None:
        eval_k_list = DEFAULT_EVAL_K_LIST

    print("=" * 70)
    print("BƯỚC 5.8: ĐÁNH GIÁ THỰC NGHIỆM BM25 TRÊN HOLD-OUT TEST SET")
    print("=" * 70)

    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)

    # 1. Kiểm tra tài nguyên đầu vào
    if not test_path.exists():
        print(f"[FAIL] Không tìm thấy file benchmark: {test_path}")
        return 1
    if not index_path.exists():
        print(f"[FAIL] Không tìm thấy chỉ mục BM25: {index_path}")
        return 1

    # 2. Nạp dữ liệu và retriever
    print(f"1. Nạp test set: {test_path.name}...")
    try:
        with open(test_path, "r", encoding="utf-8") as f:
            test_queries = json.load(f)
    except Exception as e:
        print(f"[FAIL] Lỗi khi đọc file test queries: {e}")
        return 1

    num_queries = len(test_queries)
    print(f"   -> Đã nạp {num_queries:,} test queries.")

    if num_queries == 0:
        print("[FAIL] Tập test queries rỗng, không thể đánh giá.")
        return 1

    print(f"2. Nạp BM25 Index: {index_path.name}...")
    start_load = time.time()
    try:
        retriever = BM25Retriever.load(index_path)
    except Exception as e:
        print(f"[FAIL] Lỗi khi nạp chỉ mục BM25: {e}")
        return 1
    print(f"   -> Nạp xong trong {time.time() - start_load:.2f}s")

    # 3. Chạy đánh giá từng query
    print(f"\n3. Chạy đánh giá Top-{top_k} trên {num_queries:,} queries...")
    per_query_metrics: List[Dict[str, float]] = []
    detailed_predictions: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []

    eval_start_time = time.time()

    for item in test_queries:
        qid = item.get("query_id")
        qtext = item.get("query_text", "")
        ground_truth = item.get("relevant_article_ids", [])

        # Đo latency cho từng query
        q_start = time.perf_counter()
        search_results = retriever.search(qtext, top_k=top_k)
        q_latency = (time.perf_counter() - q_start) * 1000
        latencies_ms.append(q_latency)

        retrieved_ids = [res.article_id for res in search_results]

        # Tính toán metric IR
        q_metrics = evaluate_query_retrieval(
            retrieved_ids=retrieved_ids,
            relevant_ids=ground_truth,
            k_list=eval_k_list,
        )
        per_query_metrics.append(q_metrics)

        # Lưu thông tin chi tiết phục vụ Error Analysis
        detailed_predictions.append(
            {
                "query_id": qid,
                "query_text": qtext,
                "ground_truth": ground_truth,
                "is_multi_label": len(ground_truth) > 1,
                "latency_ms": round(q_latency, 2),
                "metrics": q_metrics,
                "top_10_retrieved": [
                    {
                        "article_id": res.article_id,
                        "score": round(res.score, 4),
                        "rank": res.rank,
                        "is_relevant": res.article_id in set(ground_truth),
                    }
                    for res in search_results[:10]
                ],
            }
        )

    total_eval_time = max(time.time() - eval_start_time, 1e-6)

    # 4. Tổng hợp Macro Metrics & Hệ thống
    aggregated = aggregate_retrieval_metrics(per_query_metrics)
    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    throughput = num_queries / total_eval_time

    summary_results = {
        "model": model_name,
        "num_queries": num_queries,
        "evaluation_time_sec": round(total_eval_time, 2),
        "latency_ms_per_query": round(avg_latency, 2),
        "throughput_qps": round(throughput, 2),
        "metrics": {k: round(v, 4) for k, v in aggregated.items()},
    }

    # 5. Lưu artifacts qua temporary replacement
    tmp_metrics = metrics_output.with_suffix(".json.tmp")
    tmp_predictions = predictions_output.with_suffix(".json.tmp")

    try:
        with open(tmp_metrics, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, ensure_ascii=False, indent=2)

        with open(tmp_predictions, "w", encoding="utf-8") as f:
            json.dump(detailed_predictions, f, ensure_ascii=False, indent=2)

        tmp_metrics.replace(metrics_output)
        tmp_predictions.replace(predictions_output)
    finally:
        for tmp_file in [tmp_metrics, tmp_predictions]:
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)

    # 6. Nạp kết quả TF-IDF để so sánh đối đầu
    tfidf_metrics: Dict[str, float] = {}
    tfidf_latency = 0.0
    if tfidf_metrics_path.exists():
        try:
            with open(tfidf_metrics_path, "r", encoding="utf-8") as f:
                tfidf_payload = json.load(f)
                tfidf_metrics = tfidf_payload.get("metrics", {})
                tfidf_latency = tfidf_payload.get("latency_ms_per_query", 0.0)
        except Exception as e:
            print(f"[WARN] Không thể đọc baseline TF-IDF metrics: {e}")

    # 7. In bảng so sánh Head-to-Head
    print("\n" + "=" * 70)
    print("BẢNG ĐỐI ĐẦU TRỰC TIẾP: TF-IDF vs. BM25")
    print("=" * 70)
    print(f"Số lượng queries test: {num_queries:,}")
    print(f"Thời gian đánh giá   : {total_eval_time:.2f}s")
    print(f"Throughput           : {throughput:.2f} queries/s")
    print("-" * 70)
    print(
        f"{'Thông số':<16} | {'TF-IDF':<12} | {'BM25':<14} | {'Chênh lệch (Delta)':<18}"
    )
    print("-" * 70)

    if tfidf_metrics:
        delta_lat = avg_latency - tfidf_latency
        print(
            f"{'Latency (ms)':<16} | {tfidf_latency:<12.2f} | {avg_latency:<14.2f} | "
            f"{delta_lat:+.2f} ms"
        )
        print("-" * 70)

    for m in METRICS_TO_COMPARE:
        bm25_val = aggregated.get(m, 0.0)
        if m in tfidf_metrics:
            tf_val = tfidf_metrics[m]
            delta = bm25_val - tf_val
            pct = (delta / tf_val * 100) if tf_val > 0 else 0.0
            print(
                f"{m:<16} | {tf_val:<12.4f} | {bm25_val:<14.4f} | "
                f"{delta:+.4f} ({pct:+.2f}%)"
            )
        else:
            print(f"{m:<16} | {'N/A':<12} | {bm25_val:<14.4f} | {'N/A':<18}")

    print("=" * 70)
    print("Artifacts đã lưu thành công:")
    print(f"  - Summary Metrics  : {metrics_output}")
    print(f"  - Predictions Log  : {predictions_output}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Đánh giá mô hình BM25 Retriever trên Hold-out Test Set."
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=DEFAULT_TEST_PATH,
        help="Đường dẫn file retrieval_test.json.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Đường dẫn file bm25_index.pkl.",
    )
    parser.add_argument(
        "--tfidf-metrics",
        type=Path,
        default=DEFAULT_TFIDF_METRICS_PATH,
        help="Đường dẫn file tfidf_test_metrics.json để so sánh đối đầu.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_OUTPUT,
        help="Đường dẫn lưu file JSON metrics tổng hợp.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=DEFAULT_PREDICTIONS_OUTPUT,
        help="Đường dẫn lưu file JSON log dự đoán chi tiết.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Số lượng văn bản cần truy xuất tối đa cho mỗi query (mặc định: 100).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BM25 (Default: k1=1.5, b=0.75)",
        help="Tên mô hình để ghi vào summary payload.",
    )
    args = parser.parse_args()

    exit_code = evaluate_bm25(
        test_path=args.test,
        index_path=args.index,
        tfidf_metrics_path=args.tfidf_metrics,
        metrics_output=args.metrics_output,
        predictions_output=args.predictions_output,
        top_k=args.top_k,
        model_name=args.model_name,
    )
    sys.exit(exit_code)