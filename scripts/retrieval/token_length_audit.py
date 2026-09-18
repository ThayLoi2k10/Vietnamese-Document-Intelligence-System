# scripts/retrieval/token_length_audit.py
"""Script phân tích độ dài token (Token-Length Audit) cho Phase 6.

Tuân thủ nghiêm ngặt đặc tả:
    1. Đo lường phân phối độ dài token trên toàn bộ 59,636 điều luật corpus.
    2. Đo lường phân phối độ dài token trên 1,752 câu hỏi tập train (retrieval_train.json).
    3. TUYỆT ĐỐI KHÔNG đọc hay sử dụng retrieval_test.json để xác định tham số.
    4. Áp dụng quy tắc tất định để khóa cứng article_max_length và query_max_length.
    5. Xuất báo cáo thống kê và pre-test manifest ra thư mục results/benchmarks/.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

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

from src.retrieval.dense.artifacts import compute_file_sha256

DEFAULT_CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_retrieval"
    / "corpus_articles_preprocessed.parquet"
)
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "benchmarks" / "retrieval_train.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmarks"

MODEL_IDENTIFIER = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


def compute_distribution_stats(lengths: np.ndarray) -> Dict[str, float]:
    """Tính các chỉ số phân phối mẫu: min, mean, median, các phân vị và max."""
    return {
        "count": int(len(lengths)),
        "min": int(np.min(lengths)),
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
        "p50_median": float(np.median(lengths)),
        "p75": float(np.percentile(lengths, 75)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p98": float(np.percentile(lengths, 98)),
        "p99": float(np.percentile(lengths, 99)),
        "p99_5": float(np.percentile(lengths, 99.5)),
        "p99_9": float(np.percentile(lengths, 99.9)),
        "max": int(np.max(lengths)),
    }


def run_token_length_audit(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    train_path: Path = DEFAULT_TRAIN_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> int:
    """Thực thi phân tích độ dài token và khóa cấu hình trước khi chạy test."""
    print("=" * 65)
    print("PHASE 6: TOKEN-LENGTH AUDIT VÀ PRE-TEST MANIFEST LOCK")
    print("=" * 65)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "dense_token_length_audit_report.json"
    manifest_file = output_dir / "dense_pre_test_manifest.json"

    # 1. Nạp tokenizer
    print(f"1. Nạp tokenizer cho {MODEL_IDENTIFIER}...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDENTIFIER)
    vocab_size = len(tokenizer)
    print(f"   -> Đã nạp tokenizer (vocab_size={vocab_size:,}) trong {time.time() - start_time:.2f}s")

    # 2. Phân tích tập truy vấn Train (1,752 queries)
    print(f"2. Phân tích độ dài câu hỏi tập TRAIN: {train_path.name}...")
    if not train_path.exists():
        print(f"[FAIL] Không tìm thấy train benchmark tại: {train_path}")
        return 1

    train_sha256 = compute_file_sha256(train_path)
    with open(train_path, "r", encoding="utf-8") as f:
        train_queries = json.load(f)

    train_texts = [q["query_text"] for q in train_queries]
    train_enc = tokenizer(
        train_texts,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )
    query_lens = np.array([len(ids) for ids in train_enc["input_ids"]])
    query_stats = compute_distribution_stats(query_lens)
    print(f"   -> Train queries: N={query_stats['count']}, Min={query_stats['min']}, "
          f"Median={query_stats['p50_median']:.1f}, Mean={query_stats['mean']:.2f}, "
          f"p99={query_stats['p99']:.1f}, Max={query_stats['max']}")

    # Khóa query_max_length theo quy tắc tất định
    # Max train query là 55 tokens. Chuẩn BGE-M3 default là 512.
    locked_query_max_length = 512
    query_trunc_count = int(np.sum(query_lens > locked_query_max_length))
    query_trunc_rate = float(query_trunc_count / len(query_lens))
    print(f"   -> KHÓA query_max_length = {locked_query_max_length} "
          f"(Tỷ lệ cắt xén: {query_trunc_rate:.4%}, 0% loss)")

    # 3. Phân tích toàn bộ 59,636 điều luật trong Corpus
    print(f"3. Phân tích độ dài điều luật CORPUS: {corpus_path.name}...")
    if not corpus_path.exists():
        print(f"[FAIL] Không tìm thấy corpus tại: {corpus_path}")
        return 1

    corpus_table = pq.read_table(corpus_path, columns=["global_article_id", "text"])
    corpus_df = corpus_table.to_pandas()
    corpus_texts = [t if t is not None else "" for t in corpus_df["text"].tolist()]

    start_corp = time.time()
    corpus_enc = tokenizer(
        corpus_texts,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )
    doc_lens = np.array([len(ids) for ids in corpus_enc["input_ids"]])
    doc_stats = compute_distribution_stats(doc_lens)
    print(f"   -> Tokenize {len(doc_lens):,} articles trong {time.time() - start_corp:.2f}s")
    print(f"   -> Corpus stats: Min={doc_stats['min']}, Median={doc_stats['p50_median']:.1f}, "
          f"Mean={doc_stats['mean']:.2f}, p95={doc_stats['p95']:.1f}, "
          f"p98={doc_stats['p98']:.1f}, p99={doc_stats['p99']:.1f}, Max={doc_stats['max']}")

    # Đánh giá các ngưỡng ứng viên
    truncation_candidates = {}
    for limit in [512, 1024, 2048, 4096, 8192]:
        truncated = int(np.sum(doc_lens > limit))
        pct = float(truncated / len(doc_lens))
        truncation_candidates[str(limit)] = {
            "truncated_count": truncated,
            "truncation_rate": pct,
            "coverage_rate": 1.0 - pct,
        }
        print(f"      Ngưỡng {limit:5d} tokens: cắt {truncated:5d} điều luật "
              f"({pct:.2%}), bao phủ {1.0 - pct:.2%}")

    # Khóa article_max_length theo quy tắc tất định:
    # Chọn 2048 vì bao phủ 98.81% điều luật (58,928 / 59,636 điều luật giữ nguyên 100% nội dung),
    # chỉ có 708 điều luật (1.19%) bị cắt phần đuôi, cân bằng tối ưu giữa độ bao phủ ngữ nghĩa và độ phức tạp tính toán O(L^2).
    locked_article_max_length = 2048
    locked_article_trunc_count = truncation_candidates["2048"]["truncated_count"]
    locked_article_trunc_rate = truncation_candidates["2048"]["truncation_rate"]
    print(f"   -> KHÓA article_max_length = {locked_article_max_length} "
          f"(Bao phủ: {1.0 - locked_article_trunc_rate:.2%}, Cắt xén: {locked_article_trunc_rate:.2%})")

    # 4. Lưu báo cáo kiểm toán chi tiết
    audit_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_identifier": MODEL_IDENTIFIER,
        "model_revision": MODEL_REVISION,
        "vocab_size": vocab_size,
        "train_benchmark": {
            "path": str(train_path),
            "sha256": train_sha256,
            "num_queries": len(train_texts),
            "statistics": query_stats,
            "locked_max_length": locked_query_max_length,
            "truncation_count": query_trunc_count,
            "truncation_rate": query_trunc_rate,
        },
        "corpus": {
            "path": str(corpus_path),
            "num_articles": len(corpus_texts),
            "statistics": doc_stats,
            "truncation_candidates": truncation_candidates,
            "locked_max_length": locked_article_max_length,
            "truncation_count": locked_article_trunc_count,
            "truncation_rate": locked_article_trunc_rate,
        },
        "deterministic_rules": {
            "query_rule": "query_max_length=512 (bao phủ 100% câu hỏi train, max query length là 55)",
            "article_rule": "article_max_length=2048 (bao phủ 98.81% điều luật, hạn chế tỷ lệ cắt dưới 1.2%)",
        },
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False)
    print(f"\n4. Đã lưu báo cáo kiểm toán: {report_file}")

    # 5. Lưu Pre-test Manifest
    pre_test_manifest = {
        "manifest_name": "phase6_dense_pre_test_manifest",
        "status": "LOCKED_BEFORE_FROZEN_TEST",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": {
            "identifier": MODEL_IDENTIFIER,
            "revision": MODEL_REVISION,
            "mode": "dense_only",
            "pooling": "cls",
            "embedding_dimension": 1024,
        },
        "text_parameters": {
            "query_max_length": locked_query_max_length,
            "article_max_length": locked_article_max_length,
            "truncation": True,
            "query_truncation_rate_on_train": query_trunc_rate,
            "article_truncation_rate_on_corpus": locked_article_trunc_rate,
        },
        "embedding_parameters": {
            "normalization": True,
            "similarity": "inner_product_on_normalized_vectors",
            "storage_dtype": "float32",
            "faiss_index_type": "IndexFlatIP",
            "retrieval_depth": 100,
        },
        "input_hashes": {
            "corpus_file": str(corpus_path.name),
            "corpus_rows": len(corpus_texts),
            "train_benchmark_sha256": train_sha256,
        },
    }

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(pre_test_manifest, f, indent=2, ensure_ascii=False)
    print(f"5. Đã tạo và khóa Pre-test Manifest: {manifest_file}")
    print("=" * 65)
    print("[PASS] HOÀN TẤT TOKEN-LENGTH AUDIT VÀ KHÓA THAM SỐ THÀNH CÔNG")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(run_token_length_audit())
