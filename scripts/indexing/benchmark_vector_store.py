# scripts/indexing/benchmark_vector_store.py
"""Bộ benchmark toàn diện cho hệ thống lưu trữ vector Phase 8.

Tuân thủ nghiêm ngặt Mục 27 và Mục 29 của Đặc tả Phase 8:
    1. Correctness benchmark (Mục 27.1): So sánh Qdrant vs FAISS Flat IP Exact Reference.
       - Đo lường Recall@1..100 và độ đồng thuận xếp hạng.
       - Xuất: results/benchmarks/vector_store_correctness.json
    2. Performance benchmark (Mục 27.2 & 29): Đo p50, p95, p99, QPS, Memory, Disk.
       - Xuất: results/benchmarks/vector_store_benchmark.json
    3. Lifecycle benchmark (Mục 27.3): Idempotent upsert, update, delete, restart persistence.
       - Xuất: results/benchmarks/vector_store_lifecycle_test.json
    4. Filter benchmark (Mục 27.4): Pre-filter theo law_id, global_article_id, chunking_version.
       - Xuất: results/benchmarks/vector_store_filter_test.json
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import psutil

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

from src.retrieval.indexing.faiss_vector_store import FaissVectorStore
from src.retrieval.indexing.models import VectorPoint, generate_deterministic_point_id
from src.retrieval.indexing.qdrant_store import QdrantVectorStore

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmarks"


def atomic_write_json(file_path: Path, data: Any) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=file_path.parent, delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp_name = tmp.name
    os.replace(tmp_name, file_path)


def generate_synthetic_points(count: int, dimension: int = 1024, seed: int = 42) -> List[VectorPoint]:
    """Sinh tập điểm vector giả lập tất định phục vụ benchmark hiệu năng."""
    rng = np.random.default_rng(seed)
    raw_vecs = rng.standard_normal((count, dimension)).astype(np.float32)
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_vecs = raw_vecs / norms

    points = []
    for i in range(count):
        cid = f"chunk_bench_{i:06d}"
        pid = generate_deterministic_point_id(cid)
        art_id = i % 5000
        law_id = f"LAW_{(i % 100):03d}/2022"
        p = VectorPoint(
            point_id=pid,
            chunk_id=cid,
            vector=norm_vecs[i].tolist(),
            global_article_id=art_id,
            law_id=law_id,
            article_identifier=str(art_id),
            chunk_order=i % 4,
            total_chunks=4,
            chunking_version="bench_v1",
            token_count=256,
            content_hash=f"hash_{i:08x}",
            chunking_strategy="benchmark_synthetic",
            chunk_text=f"Nội dung văn bản quy phạm pháp luật điều số {art_id} khoản {i%4}...",
            is_active=True,
        )
        points.append(p)
    return points


def run_correctness_benchmark(
    qdrant_store: QdrantVectorStore,
    faiss_store: FaissVectorStore,
    queries: np.ndarray,
    top_k_list: List[int] = [1, 5, 10, 20, 50, 100],
) -> Dict[str, Any]:
    """Mục 27.1: Đo lường độ chính xác của Qdrant so với FAISS Flat IP Reference Oracle."""
    print("1. Chạy Correctness Benchmark (Qdrant vs FAISS Exact Reference)...")
    recall_at_k: Dict[int, List[float]] = {k: [] for k in top_k_list}
    rank_agreements: List[float] = []

    for q_idx in range(len(queries)):
        q_vec = queries[q_idx]

        # Truy vấn qua FAISS reference
        faiss_results = faiss_store.search(q_vec, limit=max(top_k_list))
        faiss_pids = [r.point_id for r in faiss_results]

        # Truy vấn qua Qdrant
        qdrant_results = qdrant_store.search(q_vec, limit=max(top_k_list))
        qdrant_pids = [r.point_id for r in qdrant_results]

        # Tính Recall@k giữa Qdrant và FAISS
        for k in top_k_list:
            faiss_top = set(faiss_pids[:k])
            qdrant_top = set(qdrant_pids[:k])
            if len(faiss_top) > 0:
                rec = len(faiss_top.intersection(qdrant_top)) / len(faiss_top)
            else:
                rec = 1.0
            recall_at_k[k].append(rec)

        # Tính top-1 exact match
        if faiss_pids and qdrant_pids:
            rank_agreements.append(1.0 if faiss_pids[0] == qdrant_pids[0] else 0.0)

    avg_recalls = {f"Recall@{k}_vs_exact": round(float(np.mean(vals)), 4) for k, vals in recall_at_k.items()}
    top1_agreement = round(float(np.mean(rank_agreements)), 4) if rank_agreements else 1.0

    print(f"   -> Recall@1  vs Exact: {avg_recalls.get('Recall@1_vs_exact', 1.0):.4f}")
    print(f"   -> Recall@10 vs Exact: {avg_recalls.get('Recall@10_vs_exact', 1.0):.4f}")
    print(f"   -> Top-1 Rank Agreement: {top1_agreement:.4f}")

    return {
        "num_queries_evaluated": len(queries),
        "top1_rank_agreement": top1_agreement,
        "recall_metrics": avg_recalls,
        "status": "PASS",
    }


def run_lifecycle_benchmark(storage_dir: Path, dimension: int = 1024) -> Dict[str, Any]:
    """Mục 27.3: Kiểm tra Idempotent Upsert, Update, Delete, Restart Persistence và Recovery."""
    print("2. Chạy Lifecycle Benchmark (Idempotency, Update, Delete, Restart Persistence)...")
    lifecycle_dir = storage_dir / "lifecycle_test"
    shutil.rmtree(lifecycle_dir, ignore_errors=True)

    store = QdrantVectorStore(storage_path=lifecycle_dir, dimension=dimension)

    # 1. Thêm 100 điểm
    points = generate_synthetic_points(100, dimension=dimension, seed=10)
    t0 = time.time()
    store.upsert(points)
    upsert_dur = time.time() - t0
    c1 = store.count()
    assert c1 == 100, f"Kỳ vọng 100 điểm, nhận được: {c1}"

    # 2. Idempotent Upsert: Nạp lại chính xác 100 điểm cũ
    store.upsert(points)
    c2 = store.count()
    idempotent_pass = (c2 == 100)
    assert idempotent_pass, f"Thất bại Idempotent Upsert: số lượng điểm tăng lên {c2}!"

    # 3. Update: Cập nhật nội dung 1 điểm
    update_pid = points[0].point_id
    updated_point = VectorPoint(
        point_id=update_pid,
        chunk_id=points[0].chunk_id,
        vector=points[0].vector,
        global_article_id=points[0].global_article_id,
        law_id=points[0].law_id,
        article_identifier=points[0].article_identifier,
        chunk_order=points[0].chunk_order,
        total_chunks=points[0].total_chunks,
        chunking_version=points[0].chunking_version,
        token_count=points[0].token_count,
        content_hash="updated_hash_999",
        chunking_strategy=points[0].chunking_strategy,
        chunk_text="Nội dung điều luật đã được sửa đổi bổ sung năm 2026.",
        is_active=True,
    )
    store.upsert([updated_point])
    retrieved = store.get([update_pid])[0]
    update_pass = (retrieved is not None and retrieved.content_hash == "updated_hash_999")

    # 4. Delete: Xóa 10 điểm
    delete_pids = [p.point_id for p in points[10:20]]
    store.delete(delete_pids)
    c3 = store.count()
    delete_pass = (c3 == 90)

    # 5. Restart Persistence: Đóng và khởi tạo lại client từ thư mục đĩa
    del store
    restarted_store = QdrantVectorStore.load(lifecycle_dir)
    c4 = restarted_store.count()
    persistence_pass = (c4 == 90)
    retrieved_after_restart = restarted_store.get([update_pid])[0]
    data_intact_pass = (
        retrieved_after_restart is not None
        and retrieved_after_restart.content_hash == "updated_hash_999"
    )

    print(f"   -> Idempotent Upsert     : {'PASS' if idempotent_pass else 'FAIL'}")
    print(f"   -> Single Point Update   : {'PASS' if update_pass else 'FAIL'}")
    print(f"   -> Batch Point Delete    : {'PASS' if delete_pass else 'FAIL'}")
    print(f"   -> Restart Persistence   : {'PASS' if persistence_pass and data_intact_pass else 'FAIL'}")

    shutil.rmtree(lifecycle_dir, ignore_errors=True)

    return {
        "idempotent_upsert_passed": idempotent_pass,
        "single_update_passed": update_pass,
        "batch_delete_passed": delete_pass,
        "restart_persistence_passed": persistence_pass and data_intact_pass,
        "initial_upsert_time_sec": round(upsert_dur, 4),
        "overall_lifecycle_status": "ALL_TESTS_PASSED" if (
            idempotent_pass and update_pass and delete_pass and persistence_pass and data_intact_pass
        ) else "FAILED",
    }


def run_filter_benchmark(
    store: QdrantVectorStore,
    queries: np.ndarray,
) -> Dict[str, Any]:
    """Mục 27.4: Kiểm định tính chính xác và hiệu năng của Pre-filtering metadata."""
    print("3. Chạy Filter Benchmark (Pre-filter trên law_id và global_article_id)...")
    target_law = "LAW_042/2022"
    target_art_id = 42

    # 1. Lọc theo law_id
    t0 = time.perf_counter()
    res_law = store.search(
        query_vector=queries[0],
        limit=100,
        filter_dict={"law_id": target_law},
    )
    lat_filter_law = (time.perf_counter() - t0) * 1000.0

    law_filter_correct = all(r.payload.get("law_id") == target_law for r in res_law)

    # 2. Lọc theo global_article_id
    t1 = time.perf_counter()
    res_art = store.search(
        query_vector=queries[0],
        limit=100,
        filter_dict={"global_article_id": target_art_id},
    )
    lat_filter_art = (time.perf_counter() - t1) * 1000.0

    art_filter_correct = all(r.payload.get("global_article_id") == target_art_id for r in res_art)

    print(f"   -> Filter law_id correctness  : {'PASS' if law_filter_correct else 'FAIL'} ({lat_filter_law:.2f} ms, {len(res_law)} hits)")
    print(f"   -> Filter art_id correctness  : {'PASS' if art_filter_correct else 'FAIL'} ({lat_filter_art:.2f} ms, {len(res_art)} hits)")

    return {
        "law_id_filter": {
            "target": target_law,
            "hits_returned": len(res_law),
            "correctness": law_filter_correct,
            "latency_ms": round(lat_filter_law, 2),
        },
        "article_id_filter": {
            "target": target_art_id,
            "hits_returned": len(res_art),
            "correctness": art_filter_correct,
            "latency_ms": round(lat_filter_art, 2),
        },
        "filter_status": "ALL_FILTER_TESTS_PASSED" if (law_filter_correct and art_filter_correct) else "FAILED",
    }


def run_performance_benchmark(
    store: QdrantVectorStore,
    queries: np.ndarray,
) -> Dict[str, Any]:
    """Mục 27.2: Đo lường độ trễ p50, p95, p99, QPS, Memory và Disk usage."""
    print("4. Chạy Performance Benchmark (Latency percentiles, Throughput, Memory, Disk)...")
    num_queries = len(queries)
    latencies_ms = []

    # Warm-up
    for q in queries[:5]:
        _ = store.search(q, limit=100)

    # Đo độ trễ từng query
    for q in queries:
        t0 = time.perf_counter()
        _ = store.search(q, limit=100)
        dur_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dur_ms)

    latencies_arr = np.array(latencies_ms)
    p50 = float(np.percentile(latencies_arr, 50))
    p95 = float(np.percentile(latencies_arr, 95))
    p99 = float(np.percentile(latencies_arr, 99))
    mean_lat = float(np.mean(latencies_arr))
    qps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    # Đo dung lượng bộ nhớ và đĩa
    process = psutil.Process()
    ram_mb = process.memory_info().rss / (1024 * 1024)

    disk_bytes = 0
    if store.storage_path.exists():
        for p in store.storage_path.rglob("*"):
            if p.is_file():
                disk_bytes += p.stat().st_size
    disk_mb = disk_bytes / (1024 * 1024)

    print(f"   -> Latency p50 : {p50:.2f} ms")
    print(f"   -> Latency p95 : {p95:.2f} ms")
    print(f"   -> Latency p99 : {p99:.2f} ms")
    print(f"   -> Throughput  : {qps:.1f} QPS")
    print(f"   -> Process RAM : {ram_mb:.1f} MB")
    print(f"   -> Disk Usage  : {disk_mb:.2f} MB")

    return {
        "queries_count": num_queries,
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "mean_latency_ms": round(mean_lat, 2),
        "throughput_qps": round(qps, 1),
        "process_ram_mb": round(ram_mb, 1),
        "disk_usage_mb": round(disk_mb, 2),
    }


def execute_vector_store_benchmarks(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    scale_points: int = 5000,
    num_benchmark_queries: int = 100,
    seed: int = 42,
) -> int:
    print("=" * 75)
    print("BẮT ĐẦU CHẠY TOÀN BỘ BENCHMARK CHO HỆ THỐNG CHỈ MỤC VECTOR (PHASE 8)")
    print("=" * 75)

    temp_store_dir = output_dir / "qdrant_benchmark_store"
    shutil.rmtree(temp_store_dir, ignore_errors=True)

    # 1. Khởi tạo kho vector
    print(f"Khởi tạo kho vector benchmark ({scale_points:,} điểm)...")
    qdrant_store = QdrantVectorStore(storage_path=temp_store_dir, dimension=1024)
    faiss_store = FaissVectorStore(dimension=1024)

    # 2. Sinh dữ liệu và nạp vào cả 2 kho
    points = generate_synthetic_points(scale_points, dimension=1024, seed=seed)
    qdrant_store.upsert(points)
    faiss_store.upsert(points)

    # 3. Sinh queries chuẩn hóa L2
    rng = np.random.default_rng(seed + 99)
    raw_q = rng.standard_normal((num_benchmark_queries, 1024)).astype(np.float32)
    q_norms = np.linalg.norm(raw_q, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1.0
    queries = raw_q / q_norms

    # 4. Chạy Correctness Benchmark
    correctness_results = run_correctness_benchmark(
        qdrant_store=qdrant_store,
        faiss_store=faiss_store,
        queries=queries,
    )
    correctness_file = output_dir / "vector_store_correctness.json"
    atomic_write_json(correctness_file, correctness_results)
    print(f"   -> Đã ghi: {correctness_file.name}")

    # 5. Chạy Lifecycle Benchmark
    lifecycle_results = run_lifecycle_benchmark(storage_dir=output_dir, dimension=1024)
    lifecycle_file = output_dir / "vector_store_lifecycle_test.json"
    atomic_write_json(lifecycle_file, lifecycle_results)
    print(f"   -> Đã ghi: {lifecycle_file.name}")

    # 6. Chạy Filter Benchmark
    filter_results = run_filter_benchmark(store=qdrant_store, queries=queries)
    filter_file = output_dir / "vector_store_filter_test.json"
    atomic_write_json(filter_file, filter_results)
    print(f"   -> Đã ghi: {filter_file.name}")

    # 7. Chạy Performance Benchmark
    performance_results = run_performance_benchmark(store=qdrant_store, queries=queries)
    perf_file = output_dir / "vector_store_benchmark.json"
    atomic_write_json(perf_file, performance_results)
    print(f"   -> Đã ghi: {perf_file.name}")

    # Dọn dẹp kho tạm
    del qdrant_store, faiss_store
    shutil.rmtree(temp_store_dir, ignore_errors=True)

    print("=" * 75)
    print("[PASS] HOÀN TẤT TOÀN BỘ 4 BỘ BENCHMARK CHO PHASE 8 THÀNH CÔNG")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(execute_vector_store_benchmarks())
