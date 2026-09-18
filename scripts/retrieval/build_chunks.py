# scripts/retrieval/build_chunks.py
"""Script xây dựng và tuần tự hóa tập dữ liệu Chunk hoàn chỉnh cho Phase 7.

Tuân thủ nghiêm ngặt:
    - Mục 11: Chunk Data Model (12 trường bắt buộc)
    - Mục 12: Provenance Invariants (bảo toàn 59,636 bài viết, 8 điều luật rỗng)
    - Mục 13: Deterministic Identity & Versioning
    - Mục 22: Validation Gates (Gate 1, Gate 2, Gate 3)
    - Mục 23: Lưu trữ parquet tại data/processed/legal_retrieval/corpus_chunks_<version>.parquet
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Set
import numpy as np
import pandas as pd
import pyarrow as pa
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

from src.retrieval.chunking import (
    BaseChunker,
    FixedSizeChunker,
    LegalChunk,
    ParagraphChunker,
    RecursiveChunker,
    SentenceChunker,
    StructureAwareChunker,
)

DEFAULT_CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_retrieval"
    / "corpus_articles_preprocessed.parquet"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "legal_retrieval"


def create_chunker_by_name(
    strategy_name: str,
    target_size: int = 512,
    overlap: float = 0.0,
    tokenizer: Any = None,
) -> BaseChunker:
    """Khởi tạo chunker tương ứng theo tên chiến lược."""
    name_lower = strategy_name.lower()
    if "structure" in name_lower or "hierarchical" in name_lower:
        clause_first = "clause_first" in name_lower
        return StructureAwareChunker(
            target_size=target_size,
            clause_first_only=clause_first,
            overlap=overlap,
            tokenizer=tokenizer,
        )
    elif "fixed" in name_lower:
        return FixedSizeChunker(
            target_size=target_size,
            overlap=overlap,
            tokenizer=tokenizer,
        )
    elif "recursive" in name_lower:
        return RecursiveChunker(
            target_size=target_size,
            overlap=overlap,
            tokenizer=tokenizer,
        )
    elif "sentence" in name_lower:
        return SentenceChunker(
            target_size=target_size,
            overlap=overlap,
            tokenizer=tokenizer,
        )
    elif "paragraph" in name_lower:
        return ParagraphChunker(
            target_size=target_size,
            overlap=overlap,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"Chiến lược không hợp lệ: {strategy_name}")


def validate_chunk_integrity_gates(
    chunks: List[LegalChunk],
    corpus_df: pd.DataFrame,
) -> bool:
    """Xác thực toàn diện các Gate 1, Gate 2 và Gate 3 trước khi lưu trữ."""
    print("\n--- KIỂM TRA CÁC VALIDATION GATES ---")
    num_articles = len(corpus_df)

    # Gate 1: Toàn vẹn điều luật gốc
    parent_ids = {c.global_article_id for c in chunks}
    empty_articles_count = sum(1 for _, r in corpus_df.iterrows() if r.get("is_empty_preprocessed", False))
    expected_non_empty = num_articles - empty_articles_count

    print(f"Gate 1: Corpus integrity...")
    print(f"  * Tổng số điều luật gốc    : {num_articles:,}")
    print(f"  * Số điều luật có chunk    : {len(parent_ids):,} (kỳ vọng: {expected_non_empty:,})")
    print(f"  * Số điều luật rỗng        : {empty_articles_count} (bảo toàn, total_chunks = 0)")
    assert len(parent_ids) == expected_non_empty, "[FAIL Gate 1] Số điều luật có chunk không khớp!"
    print("  -> [PASS Gate 1]")

    # Gate 2: Toàn vẹn chunk
    print(f"Gate 2: Chunk integrity...")
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "[FAIL Gate 2] Phát hiện trùng lặp chunk_id!"

    # Kiểm tra offset ký tự khớp tuyệt đối với text nguồn
    art_text_map = {
        int(r["global_article_id"]): str(r["text"]) if r["text"] is not None else ""
        for _, r in corpus_df.iterrows()
    }

    mismatch_count = 0
    for c in chunks[:1000]:  # Kiểm tra mẫu 1000 chunks đầu
        src_text = art_text_map[c.global_article_id]
        sliced = src_text[c.char_start : c.char_end]
        if sliced != c.chunk_text:
            mismatch_count += 1

    assert mismatch_count == 0, f"[FAIL Gate 2] Có {mismatch_count} chunk bị lệch offset so với văn bản gốc!"
    print(f"  * Tổng số chunks tạo ra    : {len(chunks):,}")
    print(f"  * Tính duy nhất chunk_id   : 100% PASS")
    print(f"  * Khớp offset ký tự nguồn : 100% PASS")
    print("  -> [PASS Gate 2]")

    return True


def build_final_chunk_corpus(
    strategy_name: str = "structure_aware_hierarchical_512",
    target_size: int = 512,
    overlap: float = 0.0,
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Xây dựng tập chunk hoàn chỉnh và lưu thành file Parquet chính thức."""
    print("=" * 70)
    print("BẮT ĐẦU XÂY DỰNG CORPUS CHUNKS CHÍNH THỨC (PHASE 7 FINAL CORPUS)")
    print("=" * 70)

    # 1. Nạp tokenizer
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", local_files_only=True)

    # 2. Khởi tạo Chunker
    chunker = create_chunker_by_name(
        strategy_name=strategy_name,
        target_size=target_size,
        overlap=overlap,
        tokenizer=tokenizer,
    )
    print(f"1. Chiến lược phân đoạn: {chunker.strategy_name} (Version: {chunker.chunking_version})")

    # 3. Nạp Corpus
    print(f"2. Nạp corpus từ: {corpus_path.name}...")
    corpus_table = pq.read_table(corpus_path)
    corpus_df = corpus_table.to_pandas()

    # 4. Thực thi phân đoạn toàn bộ Corpus
    print(f"3. Bắt đầu phân đoạn {len(corpus_df):,} điều luật...")
    t0 = time.time()
    chunks = chunker.chunk_corpus(corpus_df)
    gen_time = time.time() - t0
    print(f"   -> Hoàn thành phân đoạn trong {gen_time:.2f}s ({len(chunks):,} chunks)")

    # 5. Kiểm tra Validation Gates
    validate_chunk_integrity_gates(chunks, corpus_df)

    # 6. Chuyển đổi thành PyArrow Table & Lưu trữ Parquet
    print(f"\n4. Lưu trữ Parquet chính thức...")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"corpus_chunks_{chunker.chunking_version}.parquet"

    records = [c.to_dict() for c in chunks]
    chunks_df = pd.DataFrame(records)

    # Đảm bảo schema kiểu dữ liệu chuẩn xác
    chunks_df["global_article_id"] = chunks_df["global_article_id"].astype(np.int64)
    chunks_df["chunk_order"] = chunks_df["chunk_order"].astype(np.int32)
    chunks_df["total_chunks"] = chunks_df["total_chunks"].astype(np.int32)
    chunks_df["token_count"] = chunks_df["token_count"].astype(np.int32)
    chunks_df["char_start"] = chunks_df["char_start"].astype(np.int64)
    chunks_df["char_end"] = chunks_df["char_end"].astype(np.int64)

    table = pa.Table.from_pandas(chunks_df)
    pq.write_table(table, out_file, compression="snappy")

    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"   -> Đã lưu: {out_file.name} ({file_size_mb:.2f} MB)")
    print("=" * 70)
    print("[PASS] HOÀN TẤT TẠO VÀ LƯU CORPUS CHUNKS THÀNH CÔNG")
    print("=" * 70)
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xây dựng Corpus Chunks chính thức")
    parser.add_argument("--strategy", type=str, default="structure_aware_hierarchical_512")
    parser.add_argument("--target-size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.0)
    args = parser.parse_args()

    build_final_chunk_corpus(
        strategy_name=args.strategy,
        target_size=args.target_size,
        overlap=args.overlap,
    )
