# scripts/evaluation/evaluate_bm25_tuned.py
"""Script đánh giá thực nghiệm mô hình BM25 Tuned trên Hold-out Test Set.

Thực hiện:
    1. Tự động đọc tham số tối ưu (k1*, b*) từ file bm25_tuning_train_results.json,
       hoặc nhận tham số chỉ định trực tiếp từ CLI
    2. Nạp hold-out test benchmark (retrieval_test.json) và BM25 index
    3. Guard chống tập test rỗng (ZeroDivisionError protection)
    4. Cập nhật retriever với bộ tham số tối ưu và truy vấn Top-K
    5. Tính toán hệ thống metrics IR chuẩn (P@k, R@k, MRR@k, MAP@k, nDCG@k)
    6. So sánh đối chuẩn 3 bên: TF-IDF vs. BM25 Default vs. BM25 Tuned
    7. Ghi artifacts an toàn qua cơ chế temporary files và atomic replace
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmarks"

DEFAULT_TUNING_REPORT = DEFAULT_OUTPUT_DIR / "bm25_tuning_train_results.json"
DEFAULT_TFIDF_METRICS_PATH = DEFAULT_OUTPUT_DIR / "tfidf_test_metrics.json"
DEFAULT_BM25_DEF_METRICS_PATH = DEFAULT_OUTPUT_DIR / "bm25_default_test_metrics.json"

DEFAULT_METRICS_OUTPUT = DEFAULT_OUTPUT_DIR / "bm25_tuned_test_metrics.json"
DEFAULT_PREDICTIONS_OUTPUT = DEFAULT_OUTPUT_DIR / "bm25_tuned_test_predictions.json"

FALLBACK_K1 = 1.80
FALLBACK_B = 0.85
DEFAULT_TOP_K = 100
DEFAULT_EVAL_K_LIST = [1, 5, 10, 20, 50, 100]

METRICS_TO_COMPARE = [
    "P@1", "P@5", "P@10",
    "R@1", "R@5", "R@10", "R@20", "R@50", "R@100",
    "MRR@10", "MAP@10", "MAP@100", "nDCG@10", "nDCG@100"
]


def resolve_tuned_parameters(
    tuning_report_path: Path,
    k1_cli: Optional[float],
    b_cli: Optional[float],
) -> Tuple[float, float, str]:
    """Xác định tham số k1, b từ CLI, từ báo cáo tuning, hoặc fallback mặc định."""
    if k1_cli is not None and b_cli is not None:
        return k1_cli, b_cli, "CLI arguments"

    if tuning_report_path.exists():
        try:
            with open(tuning_report_path, "r", encoding="utf-8") as f:
                tuning_data = json.load(f)
            best_config = tuning_data.get("best_config", {})
            k1_file = best_config.get("k1")
            b_file = best_config.get("b")
            if k1_file is not None and b_file is not None:
                return float(k1_file), float(b_file), f"Auto-loaded from {tuning_report_path.name}"
        except Exception as e:
            print(f"[WARN] Không thể đọc tuning report: {e}")

    return FALLBACK_K1, FALLBACK_B, "Fallback defaults"


def evaluate_bm25_tuned(
    test_path: Path = DEFAULT_TEST_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
    tuning_report: Path = DEFAULT_TUNING_REPORT,
    tfidf_metrics_path: Path = DEFAULT_TFIDF_METRICS_PATH,
    bm25_def_metrics_path: Path = DEFAULT_BM25_DEF_METRICS_PATH,
    metrics_output: Path = DEFAULT_METRICS_OUTPUT,
    predictions_output: Path = DEFAULT_PREDICTIONS_OUTPUT,
    k1: Optional[float] = None,
    b: Optional[float] = None,
    top_k: int = DEFAULT_TOP_K,
    eval_k_list: Optional[List[int]] = None,
) -> int:
    """Thực thi đánh giá thực nghiệm BM25 Tuned trên hold-out test set."""
    if eval_k_list is None:
        eval_k_list = DEFAULT_EVAL_K_LIST

    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)

    # 1. Xác định tham số k1*, b*
    tuned_k1, tuned_b, param_source = resolve_tuned_parameters(tuning_report, k1, b)

    print("=" * 75)
    print(f"BƯỚC 5.10: ĐÁNH GIÁ BM25 TUNED (k1={tuned_k1:.2f}, b={tuned_b:.2f}) TRÊN TEST SET")
    print(f"Nguồn tham số: {param_source}")
    print("=" * 75)

    # 2. Kiểm tra tài nguyên đầu vào
    if not test_path.exists():
        print(f"[FAIL] Không tìm thấy file benchmark: {test_path}")
        return 1
    if not index_path.exists():
        print(f"[FAIL] Không tìm thấy BM25 index: {index_path}")
        return 1

    # 3. Nạp test set
    print(f"1. Nạp test set: {test_path.name}...")
    try:
        with open(test_path, "r", encoding="utf-8") as f:
            test_queries = json.load(f)
    except Exception as e:
        print(f"[FAIL] Lỗi khi đọc file test queries: {e}")
        return 1

    num_queries = len(test_queries)
    print(f"   -> Đã nạp {num_queries:,} queries.")

    if num_queries == 0:
        print("[FAIL] Tập test queries rỗng, không thể đánh giá.")
        return 1

    # 4. Nạp retriever và cập nhật tham số
    print(f"2. Nạp BM25 Index và cấu hình tham số...")
    start_load = time.time()
    try:
        retriever = BM25Retriever.load(index_path)
    except Exception as e:
        print(f"[FAIL] Lỗi khi nạp chỉ mục BM25: {e}")
        return 1

    retriever.k1 = tuned_k1
    retriever.b = tuned_b
    print(
        f"   -> Nạp và cập nhật tham số hoàn tất trong {time.time() - start_load:.2f}s "
        f"(k1={retriever.k1}, b={retriever.b})"
    )

    # 5. Chạy đánh giá từng query
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

    # 6. Tổng hợp Macro Metrics & Hệ thống
    aggregated = aggregate_retrieval_metrics(per_query_metrics)
    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    throughput = num_queries / total_eval_time

    summary_results = {
        "model": f"BM25 Tuned (k1={tuned_k1:.2f}, b={tuned_b:.2f})",
        "parameters": {"k1": tuned_k1, "b": tuned_b, "source": param_source},
        "num_queries": num_queries,
        "evaluation_time_sec": round(total_eval_time, 2),
        "latency_ms_per_query": round(avg_latency, 2),
        "throughput_qps": round(throughput, 2),
        "metrics": {k: round(v, 4) for k, v in aggregated.items()},
    }

    # 7. Lưu artifacts qua temporary replacement
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

    # 8. Nạp kết quả đối chiếu
    tfidf_metrics: Dict[str, float] = {}
    bm25_def_metrics: Dict[str, float] = {}
    tfidf_lat, bm25_def_lat = 0.0, 0.0

    if tfidf_metrics_path.exists():
        try:
            with open(tfidf_metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                tfidf_metrics = data.get("metrics", {})
                tfidf_lat = data.get("latency_ms_per_query", 0.0)
        except Exception as e:
            print(f"[WARN] Không thể nạp baseline TF-IDF: {e}")

    if bm25_def_metrics_path.exists():
        try:
            with open(bm25_def_metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                bm25_def_metrics = data.get("metrics", {})
                bm25_def_lat = data.get("latency_ms_per_query", 0.0)
        except Exception as e:
            print(f"[WARN] Không thể nạp baseline BM25 Default: {e}")

    # 9. Bảng so sánh đối chuẩn 3 bên
    print("\n" + "=" * 75)
    print("BẢNG SO SÁNH ĐỐI CHUẨN 3 BÊN: TF-IDF vs. BM25 DEFAULT vs. BM25 TUNED")
    print("=" * 75)
    print(f"{'Thông số':<12} | {'TF-IDF':<10} | {'BM25 Def':<10} | {'BM25 Tuned':<12} | {'Delta (vs Def)':<15}")
    print("-" * 75)
    delta_lat = avg_latency - bm25_def_lat if bm25_def_lat > 0 else 0.0
    print(
        f"{'Latency(ms)':<12} | {tfidf_lat:<10.2f} | {bm25_def_lat:<10.2f} | "
        f"{avg_latency:<12.2f} | {delta_lat:+.2f} ms"
    )
    print("-" * 75)

    for m in METRICS_TO_COMPARE:
        v_tf = tfidf_metrics.get(m, 0.0)
        v_def = bm25_def_metrics.get(m, 0.0)
        v_tuned = aggregated.get(m, 0.0)
        delta = v_tuned - v_def
        pct = (delta / v_def * 100) if v_def > 0 else 0.0
        print(f"{m:<12} | {v_tf:<10.4f} | {v_def:<10.4f} | {v_tuned:<12.4f} | {delta:+.4f} ({pct:+.2f}%)")

    print("=" * 75)
    print("Artifacts đã lưu thành công:")
    print(f"  - Metrics Output     : {metrics_output}")
    print(f"  - Predictions Output : {predictions_output}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Đánh giá mô hình BM25 Tuned trên Hold-out Test Set."
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
        "--tuning-report",
        type=Path,
        default=DEFAULT_TUNING_REPORT,
        help="Đường dẫn file kết quả tuning JSON để tự động trích xuất k1*, b*.",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=None,
        help="Ghi đè siêu tham số k1 (nếu không truyền sẽ tự động đọc từ tuning report).",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=None,
        help="Ghi đè siêu tham số b (nếu không truyền sẽ tự động đọc từ tuning report).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Số lượng tài liệu truy xuất tối đa cho mỗi query (mặc định: 100).",
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
    args = parser.parse_args()

    exit_code = evaluate_bm25_tuned(
        test_path=args.test,
        index_path=args.index,
        tuning_report=args.tuning_report,
        metrics_output=args.metrics_output,
        predictions_output=args.predictions_output,
        k1=args.k1,
        b=args.b,
        top_k=args.top_k,
    )
    sys.exit(exit_code)