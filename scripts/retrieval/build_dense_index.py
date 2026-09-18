# scripts/retrieval/build_dense_index.py
"""Script xây dựng và tuần tự hóa chỉ mục Dense Retrieval (BGE-M3 + FAISS Flat IP).

Thực hiện:
    1. Kiểm tra idempotency (bỏ qua nếu index đã tồn tại và hợp lệ, trừ khi có cờ --force).
    2. Nạp và kiểm định tính toàn vẹn 100% của corpus (59,636 điều luật, IDs 0..59,635).
    3. Tính mã băm định danh corpus fingerprint SHA-256.
    4. Mã hóa toàn bộ corpus sang ma trận float32 [59636, 1024] bằng BAAI/bge-m3.
       - Hỗ trợ cơ chế checkpoint từng chunk an toàn, có thể resume khi bị ngắt quãng.
       - Tối ưu hóa dynamic batching theo độ dài văn bản để tối đa hóa thông lượng GPU.
    5. Xây dựng chỉ mục FAISS IndexFlatIP (chính xác, không dùng xấp xỉ ANN).
    6. Lưu trữ nguyên tử (atomic write) 6 artifacts bắt buộc.
    7. Chạy sanity test truy vấn mẫu để nghiệm thu chất lượng xếp hạng.
    8. Xác thực toàn vẹn bằng verify_artifacts() trước khi kết thúc.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pyarrow.parquet as pq
import faiss
import torch
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

from src.retrieval.dense.artifacts import (
    EXPECTED_CORPUS_SIZE,
    EXPECTED_DIMENSION,
    DenseArtifactManager,
    compute_corpus_fingerprint,
    compute_id_sequence_fingerprint,
)
from src.retrieval.dense.encoder import (
    DEFAULT_ARTICLE_MAX_LENGTH,
    DEFAULT_QUERY_MAX_LENGTH,
    OFFICIAL_MODEL_NAME,
    OFFICIAL_MODEL_REVISION,
    DenseEncoder,
)
from src.retrieval.dense.index import DenseIndex

DEFAULT_CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_retrieval"
    / "corpus_articles_preprocessed.parquet"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "indices" / "dense_bge_m3"


def encode_chunk_bucketed(
    encoder: DenseEncoder,
    chunk_texts: List[str],
    tokenizer: Any,
    max_length: int = 2048,
) -> np.ndarray:
    """Mã hóa một chunk văn bản với kỹ thuật bucketing độ dài để tối ưu GPU và tránh OOM."""
    n = len(chunk_texts)
    chunk_embeddings = np.zeros((n, EXPECTED_DIMENSION), dtype=np.float32)

    # Ước lượng độ dài token nhanh để phân nhóm
    token_lens = [len(ids) for ids in tokenizer(
        chunk_texts,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        verbose=False,
    )["input_ids"]]

    # Phân thành 4 bucket tối ưu theo kích thước bộ nhớ
    buckets: Dict[str, List[int]] = {
        "short": [],    # <= 256 tokens -> batch_size 16
        "medium": [],   # 256 < len <= 512 -> batch_size 8
        "long": [],     # 512 < len <= 1024 -> batch_size 4
        "very_long": [],# > 1024 tokens -> batch_size 2
    }

    for idx, l in enumerate(token_lens):
        if l <= 256:
            buckets["short"].append(idx)
        elif l <= 512:
            buckets["medium"].append(idx)
        elif l <= 1024:
            buckets["long"].append(idx)
        else:
            buckets["very_long"].append(idx)

    batch_configs = [
        ("short", 32),
        ("medium", 16),
        ("long", 8),
        ("very_long", 4),
    ]

    for bname, bs in batch_configs:
        indices = buckets[bname]
        if not indices:
            continue
        texts_in_bucket = [chunk_texts[i] for i in indices]
        # Sử dụng model.encode trực tiếp với batch_size thích hợp
        out = encoder.model.encode(
            texts_in_bucket,
            batch_size=bs,
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vecs = out["dense_vecs"].astype(np.float32)
        for i, original_idx in enumerate(indices):
            chunk_embeddings[original_idx] = vecs[i]

    # Kiểm tra chuẩn L2 của từng vector và reject zero vector
    norms = np.linalg.norm(chunk_embeddings, axis=1)
    if np.any(norms == 0.0) or np.any(np.isnan(norms)):
        raise ValueError("Phát hiện vector 0 hoặc NaN trong chunk embeddings.")

    return chunk_embeddings


def build_dense_index(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    article_max_length: int = DEFAULT_ARTICLE_MAX_LENGTH,
    query_max_length: int = DEFAULT_QUERY_MAX_LENGTH,
    chunk_size: int = 2000,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    use_fp16: bool = True if torch.cuda.is_available() else False,
    force: bool = False,
) -> int:
    """Xây dựng và lưu trữ chỉ mục Dense Retrieval."""
    print("=" * 65)
    print("BƯỚC 6.1: XÂY DỰNG CHỈ MỤC DENSE RETRIEVAL (BGE-M3 + FAISS FLAT IP)")
    print("=" * 65)

    if not corpus_path.exists():
        print(f"[FAIL] Không tìm thấy corpus tại: {corpus_path}")
        return 1

    artifact_mgr = DenseArtifactManager(output_dir)

    # 1. Kiểm tra Idempotency
    if not force:
        try:
            ver_res = artifact_mgr.verify_artifacts(expected_count=EXPECTED_CORPUS_SIZE)
            print(f"Chỉ mục Dense đã tồn tại và hợp lệ tại: {output_dir}\n"
                  f"Số lượng vector: {ver_res['corpus_ids_count']:,} | Kích thước: {ver_res['embeddings_shape']}.\n"
                  "Bỏ qua bước tạo lại (Dùng cờ --force nếu muốn rebuild từ đầu).")
            return 0
        except Exception:
            # Nếu chưa có hoặc artifact chưa hoàn chỉnh thì tiếp tục xây dựng
            pass

    # 2. Đọc Corpus
    print(f"1. Đọc corpus: {corpus_path.name}...")
    start_load = time.time()
    try:
        corpus_table = pq.read_table(
            corpus_path, columns=["global_article_id", "text"]
        )
        corpus_df = corpus_table.to_pandas()
    except Exception as e:
        print(f"[FAIL] Lỗi nạp corpus: {e}")
        return 1

    load_duration = time.time() - start_load
    num_rows = len(corpus_df)
    print(f"   -> Đã nạp {num_rows:,} điều luật trong {load_duration:.2f}s")

    # 3. Guard toàn vẹn Corpus
    if num_rows != EXPECTED_CORPUS_SIZE:
        print(f"[FAIL] Số hàng corpus {num_rows} != {EXPECTED_CORPUS_SIZE} kỳ vọng.")
        return 1

    corpus_ids = corpus_df["global_article_id"].astype(int).tolist()
    if min(corpus_ids) != 0 or max(corpus_ids) != EXPECTED_CORPUS_SIZE - 1:
        print(f"[FAIL] Khoảng ID [{min(corpus_ids)}, {max(corpus_ids)}] không hợp lệ.")
        return 1
    if len(set(corpus_ids)) != EXPECTED_CORPUS_SIZE:
        print("[FAIL] Phát hiện ID trùng lặp trong corpus.")
        return 1

    # 4. Tính toán Corpus Fingerprints
    print("2. Tính toán mã định danh Corpus Fingerprint (SHA-256)...")
    corpus_texts = [t if t is not None else "" for t in corpus_df["text"].tolist()]
    records = list(zip(corpus_ids, corpus_texts))
    corpus_text_sha256 = compute_corpus_fingerprint(records)
    corpus_id_sha256 = compute_id_sequence_fingerprint(corpus_ids)
    print(f"   -> Corpus Text+ID SHA-256: {corpus_text_sha256[:16]}...")
    print(f"   -> Corpus ID Seq SHA-256:  {corpus_id_sha256[:16]}...")

    # 5. Khởi tạo Encoder & Tokenizer
    print("3. Khởi tạo mô hình BAAI/bge-m3...")
    build_start_time = time.time()
    encoder = DenseEncoder(
        model_name_or_path=OFFICIAL_MODEL_NAME,
        model_revision=OFFICIAL_MODEL_REVISION,
        device=device,
        use_fp16=use_fp16,
        query_max_length=query_max_length,
        article_max_length=article_max_length,
    )
    tokenizer = AutoTokenizer.from_pretrained(OFFICIAL_MODEL_NAME)
    print(f"   -> Thiết bị suy luận: {device} | use_fp16: {use_fp16}")

    # 6. Mã hóa theo từng chunk có cơ chế Checkpointing
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    num_chunks = (num_rows + chunk_size - 1) // chunk_size
    print(f"4. Bắt đầu mã hóa {num_rows:,} điều luật thành {num_chunks} chunks (chunk_size={chunk_size})...")

    chunk_files: List[Path] = []
    for chunk_idx in range(num_chunks):
        c_start = chunk_idx * chunk_size
        c_end = min(c_start + chunk_size, num_rows)
        chunk_file = checkpoint_dir / f"chunk_{chunk_idx:03d}_{c_start}_{c_end}.npy"
        chunk_files.append(chunk_file)

        if chunk_file.exists() and not force:
            print(f"   [RESUME] Chunk {chunk_idx + 1}/{num_chunks} [{c_start}:{c_end}] đã tồn tại.")
            continue

        c_texts = corpus_texts[c_start:c_end]
        t0 = time.time()
        c_embeddings = encode_chunk_bucketed(
            encoder=encoder,
            chunk_texts=c_texts,
            tokenizer=tokenizer,
            max_length=article_max_length,
        )
        t1 = time.time()
        dur = t1 - t0
        speed = len(c_texts) / dur if dur > 0 else 0

        # Lưu chunk nguyên tử
        tmp_chunk = checkpoint_dir / f"tmp_chunk_{chunk_idx:03d}.npy"
        np.save(tmp_chunk, c_embeddings)
        os.replace(tmp_chunk, chunk_file)
        print(f"   -> Đã mã hóa Chunk {chunk_idx + 1}/{num_chunks} [{c_start}:{c_end}] "
              f"trong {dur:.1f}s ({speed:.2f} articles/s)")

    # 7. Hợp nhất ma trận Embeddings
    print("5. Hợp nhất các chunks thành ma trận toàn vẹn [59,636, 1024]...")
    all_embeddings = np.zeros((num_rows, EXPECTED_DIMENSION), dtype=np.float32)
    for chunk_idx, chunk_file in enumerate(chunk_files):
        c_start = chunk_idx * chunk_size
        c_end = min(c_start + chunk_size, num_rows)
        c_mat = np.load(chunk_file)
        all_embeddings[c_start:c_end] = c_mat

    total_encode_time = time.time() - build_start_time
    print(f"   -> Hoàn thành mã hóa toàn bộ corpus trong {total_encode_time:.2f}s "
          f"({total_encode_time / 60:.2f} phút)")

    # 8. Xây dựng FAISS IndexFlatIP
    print("6. Xây dựng chỉ mục FAISS IndexFlatIP...")
    start_index = time.time()
    dense_index = DenseIndex(dimension=EXPECTED_DIMENSION)
    dense_index.add(all_embeddings, corpus_ids)
    index_duration = time.time() - start_index
    print(f"   -> Đã thêm {dense_index.ntotal:,} vector vào FAISS Flat IP trong {index_duration:.2f}s")

    # 9. Chuẩn bị metadata và báo cáo
    import importlib.metadata as im
    import platform
    import transformers

    config_data = {
        "name": "dense_bge_m3",
        "model": {
            "name": OFFICIAL_MODEL_NAME,
            "revision": OFFICIAL_MODEL_REVISION,
            "mode": "dense_only",
            "pooling": "cls",
            "embedding_dimension": EXPECTED_DIMENSION,
        },
        "text": {
            "article_max_length": article_max_length,
            "query_max_length": query_max_length,
            "truncation": True,
        },
        "embedding": {
            "normalize": True,
            "storage_dtype": "float32",
            "inference_dtype": "float16" if use_fp16 else "float32",
        },
        "index": {
            "type": "IndexFlatIP",
            "similarity": "inner_product_on_normalized_vectors",
            "top_k": 100,
        },
        "runtime": {
            "device": device,
            "use_fp16": use_fp16,
            "chunk_size": chunk_size,
        },
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "flagembedding_version": im.version("FlagEmbedding"),
            "faiss_version": im.version("faiss-cpu"),
            "numpy_version": np.__version__,
        },
    }

    corpus_fingerprint_data = {
        "corpus_rows": num_rows,
        "corpus_text_id_sha256": corpus_text_sha256,
        "corpus_id_sequence_sha256": corpus_id_sha256,
        "model_name": OFFICIAL_MODEL_NAME,
        "model_revision": OFFICIAL_MODEL_REVISION,
        "article_max_length": article_max_length,
        "query_max_length": query_max_length,
    }

    build_report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpus_rows": num_rows,
        "dimension": EXPECTED_DIMENSION,
        "total_encode_duration_sec": total_encode_time,
        "index_build_duration_sec": index_duration,
        "throughput_articles_per_sec": num_rows / total_encode_time if total_encode_time > 0 else 0,
        "device": device,
        "use_fp16": use_fp16,
        "memory_allocated_mb": (
            torch.cuda.memory_allocated() / (1024 * 1024) if "cuda" in device else 0
        ),
    }

    # 10. Lưu trữ toàn bộ artifacts an toàn
    print("7. Lưu trữ artifacts nguyên tử (Atomic Replace)...")
    artifact_mgr.save_all_artifacts(
        config=config_data,
        corpus_ids=corpus_ids,
        corpus_fingerprint=corpus_fingerprint_data,
        embeddings=all_embeddings,
        faiss_index_saver=lambda p: faiss.write_index(dense_index.index, str(p)),
        build_report=build_report_data,
    )
    print(f"   -> Đã lưu artifacts tại: {output_dir}")

    # 11. Chạy Sanity Test
    print("8. Chạy Sanity Test truy vấn mẫu...")
    sanity_query = "Quy định về thời hiệu xử phạt vi phạm hành chính"
    q_vec = encoder.encode_queries([sanity_query], batch_size=1)
    scores, ranked_ids = dense_index.search(q_vec, top_k=5)

    print(f"   -> Truy vấn: '{sanity_query}'")
    for r_idx, (art_id, sc) in enumerate(zip(ranked_ids[0], scores[0]), start=1):
        print(f"      Top-{r_idx}: article_id={art_id:5d} | score={sc:.4f}")

    if len(ranked_ids[0]) != 5 or scores[0][0] <= 0:
        print("[FAIL] Sanity test thất bại.")
        return 1

    # 12. Kiểm định toàn vẹn artifacts
    print("9. Kiểm định toàn vẹn artifacts qua verify_artifacts()...")
    artifact_mgr.verify_artifacts(expected_corpus_fingerprint=corpus_text_sha256)
    print("   [PASS] Tất cả các artifacts đã được kiểm định toàn vẹn 100%.")

    # 13. Dọn dẹp checkpoints tạm
    print("10. Dọn dẹp các checkpoint tạm...")
    for f in chunk_files:
        if f.exists():
            f.unlink()
    if checkpoint_dir.exists() and not any(checkpoint_dir.iterdir()):
        checkpoint_dir.rmdir()

    print("=" * 65)
    print("[PASS] HOÀN TẤT XÂY DỰNG CHỈ MỤC DENSE RETRIEVAL THÀNH CÔNG")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xây dựng chỉ mục Dense Retrieval (BGE-M3)")
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--article-max-length", type=int, default=DEFAULT_ARTICLE_MAX_LENGTH)
    parser.add_argument("--query-max-length", type=int, default=DEFAULT_QUERY_MAX_LENGTH)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-fp16", action="store_true", help="Vô hiệu hóa FP16")
    parser.add_argument("--force", action="store_true", help="Bắt buộc rebuild lại")

    args = parser.parse_args()
    sys.exit(
        build_dense_index(
            corpus_path=args.corpus_path,
            output_dir=args.output_dir,
            article_max_length=args.article_max_length,
            query_max_length=args.query_max_length,
            chunk_size=args.chunk_size,
            device=args.device,
            use_fp16=not args.no_fp16,
            force=args.force,
        )
    )
