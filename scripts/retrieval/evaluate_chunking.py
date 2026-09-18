# scripts/retrieval/evaluate_chunking.py
"""Script đánh giá thực nghiệm chính thức mô hình Chunk Retrieval trên Benchmark.

Tuân thủ nghiêm ngặt:
    - Mục 15: Article-Level Evaluation
    - Mục 16: Article Aggregation (max-score per article)
    - Mục 17: Search Depth (>= 250 chunks trước khi deduplicate)
    - Mục 18: Metrics chuẩn IR (P@k, R@k, MRR@10, MAP@10, MAP@100, nDCG@10, nDCG@100)
    - So sánh đối chuẩn với Phase 6 (dense_test_metrics.json)
    - Xuất artifacts:
        * results/benchmarks/chunking_test_metrics.json
        * results/benchmarks/chunking_test_predictions.json
        * results/benchmarks/chunking_vs_phase6_comparison.json
"""

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any, Dict, List, Set, Tuple
import numpy as np
import pyarrow.parquet as pq
import faiss
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

DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmarks" / "retrieval_test.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmarks"
DEFAULT_PHASE6_METRICS_PATH = DEFAULT_OUTPUT_DIR / "dense_test_metrics.json"
DEFAULT_ARTICLE_EMBEDDINGS_PATH = (
    PROJECT_ROOT / "data" / "indices" / "dense_bge_m3" / "embeddings.f32.npy"
)
DEFAULT_CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_retrieval"
    / "corpus_articles_preprocessed.parquet"
)

EVAL_K_LIST = [1, 5, 10, 20, 50, 100]
METRICS_TO_COMPARE = [
    "P@1", "P@5", "P@10",
    "R@1", "R@5", "R@10", "R@20", "R@50", "R@100",
    "MRR@10", "MAP@10", "MAP@100", "nDCG@10", "nDCG@100"
]


def atomic_write_json(file_path: Path, data: Any) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=file_path.parent, delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp_name = tmp.name
    os.replace(tmp_name, file_path)


def evaluate_chunking(
    benchmark_path: Path,
    chunks_path: Path,
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    article_embeddings_path: Path = DEFAULT_ARTICLE_EMBEDDINGS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    phase6_metrics_path: Path = DEFAULT_PHASE6_METRICS_PATH,
    chunk_search_depth: int = 300,
    top_k: int = 100,
    batch_size: int = 32,
) -> int:
    is_test_set = "test" in benchmark_path.name.lower()
    set_name = "HOLD-OUT TEST (FROZEN TEST ONCE)" if is_test_set else "DEV"

    print("=" * 75)
    print(f"BƯỚC 7.6: ĐÁNH GIÁ CHÍNH THỨC CHUNKING RETRIEVAL TRÊN TẬP {set_name}")
    print("=" * 75)

    if not benchmark_path.exists():
        print(f"[FAIL] Không tìm thấy benchmark tại: {benchmark_path}")
        return 1
    if not chunks_path.exists():
        print(f"[FAIL] Không tìm thấy chunk corpus tại: {chunks_path}")
        return 1

    # 1. Nạp benchmark
    b_sha = compute_file_sha256(benchmark_path)
    print(f"1. Nạp benchmark: {benchmark_path.name} (SHA-256: {b_sha[:16]}...)...")
    with open(benchmark_path, "r", encoding="utf-8") as f:
        queries_data = json.load(f)
    num_queries = len(queries_data)
    print(f"   -> Tổng số câu hỏi: {num_queries:,}")

    # 2. Nạp chunks
    print(f"2. Nạp Chunk Corpus: {chunks_path.name}...")
    chunks_table = pq.read_table(chunks_path)
    chunks_df = chunks_table.to_pandas()
    num_chunks = len(chunks_df)
    chunking_version = str(chunks_df["chunking_version"].iloc[0])
    chunking_strategy = str(chunks_df["chunking_strategy"].iloc[0])
    print(f"   -> Đã nạp {num_chunks:,} chunks (Chiến lược: {chunking_strategy}, Version: {chunking_version})")

    # 3. Nạp Corpus và Phase 6 Embeddings
    print("3. Nạp Corpus và embeddings tham chiếu Phase 6...")
    corpus_df = pq.read_table(corpus_path, columns=["global_article_id", "text"]).to_pandas()
    art_len_map = {
        int(r["global_article_id"]): len(str(r["text"])) if r["text"] else 0
        for _, r in corpus_df.iterrows()
    }
    article_embeddings = np.load(article_embeddings_path)

    # 4. Khởi tạo BGE-M3 Encoder
    print("4. Khởi tạo BGE-M3 Encoder...")
    t_enc_init = time.time()
    encoder = DenseEncoder()
    print(f"   -> Đã nạp encoder trong {time.time() - t_enc_init:.2f}s")

    # 5. Xây dựng Chunk Embeddings
    print("5. Xây dựng ma trận embeddings cho các chunks...")
    cache_file = output_dir / "chunk_embeddings_cache.npz"
    cache_dict = {}
    if cache_file.exists():
        try:
            loaded_cache = np.load(cache_file)
            for k in loaded_cache.files:
                cache_dict[k] = loaded_cache[k]
        except Exception:
            pass

    chunk_matrix = np.zeros((num_chunks, 1024), dtype=np.float32)
    chunk_to_art_id = chunks_df["global_article_id"].to_numpy(dtype=np.int32)
    c_hashes = chunks_df["content_hash"].tolist()
    c_texts = chunks_df["chunk_text"].tolist()
    c_total = chunks_df["total_chunks"].to_numpy()
    c_starts = chunks_df["char_start"].to_numpy()
    c_ends = chunks_df["char_end"].to_numpy()

    missing_indices = []
    missing_texts = []

    for i in range(num_chunks):
        art_id = chunk_to_art_id[i]
        # Nếu điều luật có 1 chunk trọn vẹn văn bản -> tái sử dụng trực tiếp vector Phase 6
        if c_total[i] == 1 and c_starts[i] == 0 and c_ends[i] == art_len_map.get(art_id, 0):
            chunk_matrix[i] = article_embeddings[art_id]
        elif c_hashes[i] in cache_dict:
            chunk_matrix[i] = cache_dict[c_hashes[i]]
        else:
            missing_indices.append(i)
            missing_texts.append(c_texts[i])

    if missing_indices:
        print(f"   -> Cần mã hóa {len(missing_indices):,} chunk mới...")
        t_enc = time.time()
        new_vecs = encoder.encode_articles(missing_texts, batch_size=64)
        for idx, m_idx in enumerate(missing_indices):
            chunk_matrix[m_idx] = new_vecs[idx]
            cache_dict[c_hashes[m_idx]] = new_vecs[idx]
        print(f"   -> Mã hóa hoàn tất trong {time.time() - t_enc:.2f}s")
        np.savez_compressed(cache_file, **cache_dict)

    # L2 normalize
    norms = np.linalg.norm(chunk_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    chunk_matrix /= norms

    # 6. Xây dựng FAISS FlatIP Index
    print("6. Xây dựng chỉ mục FAISS IndexFlatIP cho các chunks...")
    t_idx = time.time()
    chunk_faiss_index = faiss.IndexFlatIP(1024)
    chunk_faiss_index.add(chunk_matrix)
    print(f"   -> FAISS Index size: {chunk_faiss_index.ntotal:,} vectors trong {time.time() - t_idx:.2f}s")

    # 7. Mã hóa câu truy vấn và tìm kiếm
    print(f"7. Thực hiện truy vấn cho {num_queries:,} queries (search_depth={chunk_search_depth})...")
    q_texts = [q["query_text"] for q in queries_data]
    t_q_enc = time.perf_counter()
    query_vecs = encoder.encode_queries(q_texts, batch_size=batch_size)
    q_enc_time = time.perf_counter() - t_q_enc

    t_search = time.perf_counter()
    scores_mat, indices_mat = chunk_faiss_index.search(query_vecs, chunk_search_depth)
    search_time = time.perf_counter() - t_search

    total_time = q_enc_time + search_time
    avg_lat_ms = (total_time / num_queries) * 1000.0
    qps = num_queries / total_time if total_time > 0 else 0

    print(f"   -> Thời gian mã hóa query: {q_enc_time:.3f}s ({q_enc_time/num_queries*1000:.2f} ms/q)")
    print(f"   -> Thời gian tìm kiếm FAISS: {search_time:.3f}s ({search_time/num_queries*1000:.2f} ms/q)")
    print(f"   -> Tổng thời gian: {total_time:.3f}s | Latency: {avg_lat_ms:.2f} ms/q | QPS: {qps:.2f}")

    # 8. Max-score Article Aggregation & Metrics
    print("8. Tổng hợp điểm số điều luật (Max-score Aggregation) và tính IR metrics...")
    per_query_metrics = []
    predictions_artifact = []

    for q_idx, q in enumerate(queries_data):
        qid = q["query_id"]
        qtext = q["query_text"]
        rel_ids = set(q["relevant_article_ids"])

        row_indices = indices_mat[q_idx]
        row_scores = scores_mat[q_idx]

        article_max_score: Dict[int, float] = {}
        for c_idx, sc in zip(row_indices, row_scores):
            if c_idx < 0:
                continue
            art_id = int(chunk_to_art_id[c_idx])
            if sc > article_max_score.get(art_id, -1e9):
                article_max_score[art_id] = float(sc)

        sorted_arts = sorted(article_max_score.items(), key=lambda x: x[1], reverse=True)
        ranked_art_ids = [item[0] for item in sorted_arts[:top_k]]
        ranked_scores = [item[1] for item in sorted_arts[:top_k]]

        q_met = evaluate_query_retrieval(
            retrieved_ids=ranked_art_ids,
            relevant_ids=rel_ids,
            k_list=EVAL_K_LIST,
        )
        per_query_metrics.append(q_met)

        top10_records = []
        for r_idx, (art_id, sc) in enumerate(zip(ranked_art_ids[:10], ranked_scores[:10]), start=1):
            top10_records.append({
                "article_id": int(art_id),
                "score": round(float(sc), 4),
                "rank": r_idx,
                "is_relevant": bool(art_id in rel_ids),
            })

        predictions_artifact.append({
            "query_id": qid,
            "query_text": qtext,
            "ground_truth": sorted(list(rel_ids)),
            "is_multi_label": bool(len(rel_ids) > 1),
            "metrics": q_met,
            "top_10_retrieved": top10_records,
            "top_100_retrieved_ids": [int(x) for x in ranked_art_ids[:100]],
        })

    macro_metrics = aggregate_retrieval_metrics(per_query_metrics)

    # 9. So sánh với Phase 6
    phase6_data = None
    if phase6_metrics_path.exists():
        try:
            with open(phase6_metrics_path, "r", encoding="utf-8") as f:
                phase6_data = json.load(f)
        except Exception:
            pass

    print("\n" + "=" * 75)
    print(f"BẢNG SO SÁNH HIỆU NĂNG: PHASE 6 (ARTICLE-LEVEL) vs PHASE 7 (CHUNK-LEVEL)")
    print("=" * 75)
    print(f"{'Chỉ số':<12} | {'Phase 6 Baseline':<18} | {'Phase 7 Chunking':<18} | {'Chênh lệch (Delta)':<18}")
    print("-" * 75)

    comparison_dict = {}
    for m in METRICS_TO_COMPARE:
        chunk_val = macro_metrics.get(m, 0.0)
        p6_val = phase6_data.get("metrics", {}).get(m) if phase6_data else None

        if p6_val is not None:
            delta = chunk_val - p6_val
            pct_delta = (delta / p6_val * 100) if p6_val > 0 else 0
            delta_str = f"{delta:+.4f} ({pct_delta:+.2f}%)"
            p6_str = f"{p6_val:.4f}"
        else:
            delta_str = "N/A"
            p6_str = "N/A"

        print(f"{m:<12} | {p6_str:<18} | {chunk_val:.4f}             | {delta_str}")
        comparison_dict[m] = {
            "phase6_baseline": p6_val,
            "phase7_chunking": round(chunk_val, 4),
            "delta": round(delta, 4) if p6_val is not None else None,
        }

    print("-" * 75)
    p6_lat = phase6_data.get("timing", {}).get("latency_ms_per_query", "N/A") if phase6_data else "N/A"
    p6_lat_str = f"{p6_lat:.2f} ms" if isinstance(p6_lat, (int, float)) else str(p6_lat)
    print(f"{'Latency':<12} | {p6_lat_str:<18} | {avg_lat_ms:.2f} ms          | "
          f"{avg_lat_ms - p6_lat:+.2f} ms" if isinstance(p6_lat, (int, float)) else "")
    print("=" * 75)

    # 10. Lưu trữ các artifacts
    prefix = "chunking_test" if is_test_set else "chunking_dev"
    metrics_file = output_dir / f"{prefix}_metrics.json"
    pred_file = output_dir / f"{prefix}_predictions.json"
    comp_file = output_dir / ("chunking_vs_phase6_comparison.json" if is_test_set else f"{prefix}_vs_phase6_comparison.json")

    metrics_payload = {
        "phase": 7,
        "model": "BGE-M3 + Chunk Retrieval + Max-Score Aggregation",
        "chunking_strategy": chunking_strategy,
        "chunking_version": chunking_version,
        "total_chunks": num_chunks,
        "chunk_search_depth": chunk_search_depth,
        "aggregation": "max_score",
        "benchmark": {
            "name": benchmark_path.name,
            "sha256": b_sha,
            "num_queries": num_queries,
        },
        "timing": {
            "query_encoding_sec": round(q_enc_time, 3),
            "faiss_search_sec": round(search_time, 3),
            "total_retrieval_sec": round(total_time, 3),
            "latency_ms_per_query": round(avg_lat_ms, 2),
            "throughput_qps": round(qps, 2),
        },
        "metrics": {k: round(v, 4) for k, v in macro_metrics.items()},
    }

    print(f"\n9. Lưu trữ các artifacts:")
    atomic_write_json(metrics_file, metrics_payload)
    print(f"   -> Metrics:     {metrics_file.name}")

    atomic_write_json(pred_file, predictions_artifact)
    print(f"   -> Predictions: {pred_file.name}")

    if phase6_data:
        comp_payload = {
            "benchmark": benchmark_path.name,
            "num_queries": num_queries,
            "chunking_version": chunking_version,
            "chunking_strategy": chunking_strategy,
            "phase6_source": str(phase6_metrics_path.name),
            "comparison": comparison_dict,
        }
        atomic_write_json(comp_file, comp_payload)
        print(f"   -> Comparison:  {comp_file.name}")

    print("=" * 75)
    print("[PASS] ĐÁNH GIÁ CHUNKING RETRIEVAL HOÀN TẤT THÀNH CÔNG")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá Chunk Retrieval trên Benchmark")
    parser.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--chunks-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-depth", type=int, default=300)
    args = parser.parse_args()

    sys.exit(
        evaluate_chunking(
            benchmark_path=args.benchmark_path,
            chunks_path=args.chunks_path,
            output_dir=args.output_dir,
            chunk_search_depth=args.chunk_depth,
        )
    )
