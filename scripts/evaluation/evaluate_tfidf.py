# scripts/evaluation/evaluate_tfidf.py
"""Script đánh giá thực nghiệm mô hình TF-IDF Retriever trên Hold-out Test Set.

Thực hiện:
    1. Nạp hold-out test benchmark (retrieval_test.json) và TF-IDF index đã fit
    2. Guard chống tập test rỗng (ZeroDivisionError protection)
    3. Truy vấn Top-K cho từng test query, đo latency chính xác bằng time.perf_counter()
    4. Tính toán hệ thống metrics IR chuẩn (P@k, R@k, MRR@k, MAP@k, nDCG@k)
    5. Xuất báo cáo tổng hợp và log chi tiết phục vụ Error Analysis qua atomic replace
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

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
from src.retrieval.tfidf_retriever import TfidfRetriever


DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "benchmarks" / "retrieval_test.json"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "indices" / "tfidf_index.joblib"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmarks"
DEFAULT_METRICS_OUTPUT = DEFAULT_OUTPUT_DIR / "tfidf_test_metrics.json"
DEFAULT_PREDICTIONS_OUTPUT = DEFAULT_OUTPUT_DIR / "tfidf_test_predictions.json"

DEFAULT_TOP_K = 100
DEFAULT_EVAL_K_LIST = [1, 5, 10, 20, 50, 100]


def evaluate_tfidf(
    test_path: Path = DEFAULT_TEST_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
    metrics_output: Path = DEFAULT_METRICS_OUTPUT,
    predictions_output: Path = DEFAULT_PREDICTIONS_OUTPUT,
    top_k: int = DEFAULT_TOP_K,
    eval_k_list: List[int] = None,
) -> int:
    """Thực thi đánh giá thực nghiệm TF-IDF trên hold-out test set."""
    if eval_k_list is None:
        eval_k_list = DEFAULT_EVAL_K_LIST

    print("=" * 65)
    print("BƯỚC 5.5: ĐÁNH GIÁ THỰC NGHIỆM TF-IDF TRÊN HOLD-OUT TEST SET")
    print("=" * 65)

    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)

    # 1. Kiểm tra tài nguyên đầu vào
    if not test_path.exists():
        print(f"[FAIL] Không tìm thấy benchmark test tại: {test_path}")
        return 1
    if not index_path.exists():
        print(f"[FAIL] Không tìm thấy chỉ mục TF-IDF tại: {index_path}")
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

    print(f"2. Nạp TF-IDF Index: {index_path.name}...")
    start_load = time.time()
    try:
        retriever = TfidfRetriever.load(index_path)
    except Exception as e:
        print(f"[FAIL] Lỗi khi nạp chỉ mục TF-IDF: {e}")
        return 1
    print(f"   -> Nạp xong trong {time.time() - start_load:.2f}s")

    # 3. Chạy đánh giá từng query
    print(f"\n3. Tiến hành tìm kiếm Top-{top_k} và tính toán metrics...")
    per_query_metrics: List[Dict[str, float]] = []
    detailed_predictions: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []

    start_eval = time.time()

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

    total_eval_time = max(time.time() - start_eval, 1e-6)

    # 4. Tổng hợp Macro Metrics & Thống kê hệ thống
    aggregated = aggregate_retrieval_metrics(per_query_metrics)
    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    throughput_qps = num_queries / total_eval_time

    summary_results = {
        "model": "TF-IDF (Baseline 1A)",
        "num_queries": num_queries,
        "evaluation_time_sec": round(total_eval_time, 2),
        "latency_ms_per_query": round(avg_latency, 2),
        "throughput_qps": round(throughput_qps, 2),
        "metrics": {k: round(v, 4) for k, v in aggregated.items()},
    }

    # 5. Lưu kết quả nguyên tử qua temporary replacement
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

    # 6. In bảng kết quả chuẩn hóa
    print("\n" + "=" * 65)
    print("KẾT QUẢ THỰC NGHIỆM: TF-IDF BASELINE (HOLD-OUT TEST SET)")
    print("=" * 65)
    print(f"Số lượng queries    : {num_queries:,}")
    print(f"Thời gian đánh giá  : {total_eval_time:.2f}s")
    print(f"Độ trễ trung bình   : {avg_latency:.2f} ms/query")
    print(f"Throughput          : {throughput_qps:.2f} queries/s")
    print("-" * 65)
    print(f"{'Metric':<12} | {'Score':<10} | {'Metric':<12} | {'Score':<10}")
    print("-" * 65)
    for k_val in eval_k_list:
        p_key = f"P@{k_val}"
        r_key = f"R@{k_val}"
        p_score = aggregated.get(p_key, 0.0)
        r_score = aggregated.get(r_key, 0.0)
        print(f"{p_key:<12} | {p_score:<10.4f} | {r_key:<12} | {r_score:<10.4f}")

    print("-" * 65)
    mrr_10 = aggregated.get("MRR@10", 0.0)
    ndcg_10 = aggregated.get("nDCG@10", 0.0)
    map_10 = aggregated.get("MAP@10", 0.0)
    ndcg_100 = aggregated.get("nDCG@100", 0.0)
    map_100 = aggregated.get("MAP@100", 0.0)

    print(f"{'MRR@10':<12} | {mrr_10:<10.4f} | {'nDCG@10':<12} | {ndcg_10:<10.4f}")
    print(f"{'MAP@10':<12} | {map_10:<10.4f} | {'nDCG@100':<12} | {ndcg_100:<10.4f}")
    print(f"{'MAP@100':<12} | {map_100:<10.4f} |")
    print("=" * 65)
    print("Artifacts đã lưu thành công:")
    print(f"  - Summary Metrics    : {metrics_output}")
    print(f"  - Detailed Log (QA)  : {predictions_output}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Đánh giá mô hình TF-IDF Retriever trên Hold-out Test Set."
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
        help="Đường dẫn file tfidf_index.joblib.",
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
    args = parser.parse_args()

    exit_code = evaluate_tfidf(
        test_path=args.test,
        index_path=args.index,
        metrics_output=args.metrics_output,
        predictions_output=args.predictions_output,
        top_k=args.top_k,
    )
    sys.exit(exit_code)