# scripts/evaluation/evaluate_dense.py
"""Script đánh giá thực nghiệm mô hình Dense Retrieval (BGE-M3) trên Benchmark.

Thực hiện:
    1. Nạp benchmark dataset (retrieval_test.json hoặc retrieval_train.json).
    2. Nạp chỉ mục Dense Retrieval đã xây dựng (BGE-M3 + FAISS Flat IP).
    3. Warm-up mô hình để loại bỏ độ trễ khởi tạo GPU/CPU.
    4. Thực hiện truy vấn Top-100 cho toàn bộ câu hỏi và đo lường thời gian chi tiết.
    5. Tính toán hệ thống metrics IR chuẩn (P@k, R@k, MRR@k, MAP@k, nDCG@k).
    6. So sánh đối chuẩn trực tiếp với Baseline BM25 Tuned (Phase 5).
    7. Ghi các artifact kết quả an toàn qua cơ chế temporary files và atomic replace.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
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

from src.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    evaluate_query_retrieval,
)
from src.retrieval.dense.artifacts import compute_file_sha256
from src.retrieval.dense.encoder import DenseEncoder
from src.retrieval.dense.retriever import DenseRetriever

DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmarks" / "retrieval_test.json"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "indices" / "dense_bge_m3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmarks"
DEFAULT_BM25_METRICS_PATH = DEFAULT_OUTPUT_DIR / "bm25_tuned_test_metrics.json"

DEFAULT_TOP_K = 100
DEFAULT_EVAL_K_LIST = [1, 5, 10, 20, 50, 100]

METRICS_TO_COMPARE = [
    "P@1", "P@5", "P@10",
    "R@1", "R@5", "R@10", "R@20", "R@50", "R@100",
    "MRR@10", "MAP@10", "MAP@100", "nDCG@10", "nDCG@100"
]


def atomic_write_json(file_path: Path, data: Any) -> None:
    """Ghi an toàn dữ liệu JSON qua cơ chế temporary file và atomic replace."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=file_path.parent, delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp_name = tmp.name
    os.replace(tmp_name, file_path)


def evaluate_dense(
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
    index_dir: Path = DEFAULT_INDEX_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    bm25_metrics_path: Path = DEFAULT_BM25_METRICS_PATH,
    top_k: int = DEFAULT_TOP_K,
    batch_size: int = 32,
    warmup_count: int = 10,
) -> int:
    """Thực thi đánh giá benchmark Dense Retrieval."""
    is_test_set = "test" in benchmark_path.name.lower()
    set_name = "HOLD-OUT TEST" if is_test_set else "TRAIN / DEV"

    print("=" * 70)
    print(f"BƯỚC 6.2: ĐÁNH GIÁ THỰC NGHIỆM DENSE RETRIEVAL (BGE-M3) TRÊN TẬP {set_name}")
    print("=" * 70)

    if not benchmark_path.exists():
        print(f"[FAIL] Không tìm thấy benchmark tại: {benchmark_path}")
        return 1
    if not index_dir.exists():
        print(f"[FAIL] Không tìm thấy thư mục chỉ mục Dense tại: {index_dir}")
        return 1

    # 1. Nạp benchmark
    benchmark_sha256 = compute_file_sha256(benchmark_path)
    print(f"1. Nạp benchmark: {benchmark_path.name} (SHA-256: {benchmark_sha256[:16]}...)...")
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    num_queries = len(benchmark_data)
    print(f"   -> Tổng số câu hỏi: {num_queries:,}")
    if num_queries == 0:
        print("[FAIL] Benchmark rỗng.")
        return 1

    # 2. Nạp DenseRetriever
    print(f"2. Nạp DenseRetriever từ {index_dir}...")
    start_load = time.time()
    encoder = DenseEncoder()
    retriever = DenseRetriever.load(index_dir, encoder=encoder)
    load_duration = time.time() - start_load
    print(f"   -> Nạp mô hình và index hoàn tất trong {load_duration:.2f}s "
          f"(Index size: {retriever.index.ntotal:,} vectors)")

    # 3. Warm-up
    if warmup_count > 0:
        print(f"3. Chạy Warm-up ({warmup_count} queries để ổn định GPU pipeline)...")
        warmup_texts = [q["query_text"] for q in benchmark_data[:warmup_count]]
        _ = retriever.batch_search(warmup_texts, top_k=10, batch_size=batch_size)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("   -> Warm-up hoàn tất.")

    # 4. Thực thi truy vấn và đo lường thời gian
    print(f"4. Bắt đầu truy vấn Top-{top_k} cho {num_queries:,} queries (batch_size={batch_size})...")
    queries_text = [q["query_text"] for q in benchmark_data]

    # Đo thời gian Query Encoding
    t_enc_start = time.perf_counter()
    query_embeddings = encoder.encode_queries(queries_text, batch_size=batch_size)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_enc_duration = time.perf_counter() - t_enc_start

    # Đo thời gian FAISS Index Search
    t_search_start = time.perf_counter()
    scores_matrix, doc_ids_list = retriever.index.search(query_embeddings, top_k=top_k)
    t_search_duration = time.perf_counter() - t_search_start

    total_eval_time = t_enc_duration + t_search_duration
    latency_per_query_ms = (total_eval_time / num_queries) * 1000.0
    throughput_qps = num_queries / total_eval_time if total_eval_time > 0 else 0

    print(f"   -> Thời gian mã hóa query: {t_enc_duration:.3f}s ({t_enc_duration/num_queries*1000:.2f} ms/query)")
    print(f"   -> Thời gian tìm kiếm FAISS: {t_search_duration:.3f}s ({t_search_duration/num_queries*1000:.2f} ms/query)")
    print(f"   -> Tổng thời gian truy vấn: {total_eval_time:.3f}s")
    print(f"   -> Độ trễ trung bình: {latency_per_query_ms:.2f} ms/query | Thông lượng: {throughput_qps:.2f} QPS")

    # 5. Tính toán metrics cho từng query và tổng hợp
    print("5. Tính toán các chỉ số đánh giá chất lượng truy xuất (IR Metrics)...")
    per_query_metrics_list = []
    predictions_artifact = []

    for i, q in enumerate(benchmark_data):
        qid = q["query_id"]
        qtext = q["query_text"]
        relevant_ids = set(q["relevant_article_ids"])
        is_multi_label = len(relevant_ids) > 1

        retrieved_ids = doc_ids_list[i]
        retrieved_scores = scores_matrix[i]

        q_metrics = evaluate_query_retrieval(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            k_list=DEFAULT_EVAL_K_LIST,
        )
        per_query_metrics_list.append(q_metrics)

        # Chuẩn bị thông tin Top-10 lưu artifact phục vụ phân tích sai số
        top_10_records = []
        for rank_idx, (art_id, sc) in enumerate(zip(retrieved_ids[:10], retrieved_scores[:10]), start=1):
            top_10_records.append({
                "article_id": int(art_id),
                "score": round(float(sc), 4),
                "rank": rank_idx,
                "is_relevant": bool(art_id in relevant_ids),
            })

        predictions_artifact.append({
            "query_id": qid,
            "query_text": qtext,
            "ground_truth": sorted(list(relevant_ids)),
            "is_multi_label": is_multi_label,
            "metrics": q_metrics,
            "top_10_retrieved": top_10_records,
            "top_100_retrieved_ids": [int(x) for x in retrieved_ids[:100]],
        })

    macro_metrics = aggregate_retrieval_metrics(per_query_metrics_list)

    # 6. Đọc BM25 metrics để so sánh đối chuẩn
    bm25_data = None
    if bm25_metrics_path.exists():
        try:
            with open(bm25_metrics_path, "r", encoding="utf-8") as f:
                bm25_data = json.load(f)
        except Exception:
            pass

    # 7. In bảng so sánh đối chuẩn 2 bên
    print("\n" + "=" * 70)
    print(f"BẢNG SO SÁNH HIỆU NĂNG: BM25 TUNED vs DENSE RETRIEVAL (BGE-M3)")
    print("=" * 70)
    print(f"{'Chỉ số':<12} | {'BM25 Tuned':<14} | {'Dense BGE-M3':<14} | {'Chênh lệch (Delta)':<18}")
    print("-" * 70)

    comparison_dict = {}
    for m in METRICS_TO_COMPARE:
        dense_val = macro_metrics.get(m, 0.0)
        bm25_val = bm25_data.get("metrics", {}).get(m) if bm25_data else None

        if bm25_val is not None:
            delta = dense_val - bm25_val
            pct_delta = (delta / bm25_val * 100) if bm25_val > 0 else 0
            delta_str = f"{delta:+.4f} ({pct_delta:+.2f}%)"
            bm25_str = f"{bm25_val:.4f}"
        else:
            delta_str = "N/A"
            bm25_str = "N/A"

        print(f"{m:<12} | {bm25_str:<14} | {dense_val:.4f}        | {delta_str}")
        comparison_dict[m] = {
            "bm25_tuned": bm25_val,
            "dense_bge_m3": round(dense_val, 4),
            "delta": round(delta, 4) if bm25_val is not None else None,
        }

    print("-" * 70)
    bm25_lat = bm25_data.get("latency_ms_per_query", "N/A") if bm25_data else "N/A"
    bm25_lat_str = f"{bm25_lat:.2f} ms" if isinstance(bm25_lat, (int, float)) else str(bm25_lat)
    print(f"{'Latency':<12} | {bm25_lat_str:<14} | {latency_per_query_ms:.2f} ms      | "
          f"{latency_per_query_ms - bm25_lat:+.2f} ms" if isinstance(bm25_lat, (int, float)) else "")
    print("=" * 70)

    # 8. Lưu trữ kết quả artifacts
    output_prefix = "dense_test" if is_test_set else "dense_train"
    metrics_file = output_dir / f"{output_prefix}_metrics.json"
    predictions_file = output_dir / f"{output_prefix}_predictions.json"
    comparison_file = output_dir / f"{output_prefix}_vs_bm25_comparison.json"

    metrics_payload = {
        "model": "Dense BGE-M3 (dense-only, cls pooling, 1024-d, FlatIP)",
        "model_identifier": encoder.model_name_or_path,
        "model_revision": encoder.model_revision,
        "benchmark": {
            "name": benchmark_path.name,
            "sha256": benchmark_sha256,
            "num_queries": num_queries,
        },
        "timing": {
            "model_loading_sec": round(load_duration, 2),
            "query_encoding_sec": round(t_enc_duration, 3),
            "faiss_search_sec": round(t_search_duration, 3),
            "total_retrieval_sec": round(total_eval_time, 3),
            "latency_ms_per_query": round(latency_per_query_ms, 2),
            "throughput_qps": round(throughput_qps, 2),
        },
        "hardware": {
            "device": encoder.device,
            "use_fp16": encoder.use_fp16,
            "os": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
        },
        "metrics": {k: round(v, 4) for k, v in macro_metrics.items()},
    }

    print(f"\n6. Lưu trữ các artifacts tại {output_dir}:")
    atomic_write_json(metrics_file, metrics_payload)
    print(f"   -> Metrics:     {metrics_file.name}")

    atomic_write_json(predictions_file, predictions_artifact)
    print(f"   -> Predictions: {predictions_file.name}")

    if bm25_data:
        comparison_payload = {
            "benchmark": benchmark_path.name,
            "num_queries": num_queries,
            "bm25_source": str(bm25_metrics_path.name),
            "comparison": comparison_dict,
        }
        atomic_write_json(comparison_file, comparison_payload)
        print(f"   -> Comparison:  {comparison_file.name}")

    print("=" * 70)
    print("[PASS] ĐÁNH GIÁ THỰC NGHIỆM DENSE RETRIEVAL HOÀN TẤT THÀNH CÔNG")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Dense Retrieval trên Benchmark")
    parser.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bm25-metrics", type=Path, default=DEFAULT_BM25_METRICS_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()
    sys.exit(
        evaluate_dense(
            benchmark_path=args.benchmark_path,
            index_dir=args.index_dir,
            output_dir=args.output_dir,
            bm25_metrics_path=args.bm25_metrics,
            top_k=args.top_k,
            batch_size=args.batch_size,
        )
    )
