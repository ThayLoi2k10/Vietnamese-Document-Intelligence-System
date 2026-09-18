# scripts/indexing/build_production_qdrant_index.py
"""Script xây dựng chỉ mục cơ sở dữ liệu vector Qdrant chính thức (Production Vector Index) cho Phase 8.

Tuân thủ nghiêm ngặt đặc tả Phase 8:
    - Tiêu thụ trực tiếp tệp chunk đông băng từ Phase 7:
      data/processed/legal_retrieval/corpus_chunks_struct_hier_512_o0_v1.parquet (79,977 chunks)
    - Tái sử dụng embedding từ cache và Phase 6 dense index
    - Sinh point_id tất định theo chuẩn UUID v5 từ chunk_id
    - Lưu trữ đầy đủ vector 1024-d và payload metadata theo Mục 11
    - Lưu trữ persistent tại data/indices/qdrant_legal/
    - Thực hiện kiểm toán đối soát (reconciliation) ngay sau khi nạp
"""

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import numpy as np
import pyarrow.parquet as pq

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

from src.retrieval.indexing.models import VectorPoint, generate_deterministic_point_id
from src.retrieval.indexing.qdrant_store import QdrantVectorStore
from src.retrieval.indexing.reconciliation import run_reconciliation

DEFAULT_CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_retrieval"
    / "corpus_chunks_struct_hier_512_o0_v1.parquet"
)
DEFAULT_CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_retrieval"
    / "corpus_articles_preprocessed.parquet"
)
DEFAULT_ARTICLE_EMBEDDINGS = (
    PROJECT_ROOT / "data" / "indices" / "dense_bge_m3" / "embeddings.f32.npy"
)
DEFAULT_CACHE_FILE = (
    PROJECT_ROOT / "results" / "benchmarks" / "chunk_embeddings_cache.npz"
)
DEFAULT_QDRANT_STORAGE = PROJECT_ROOT / "data" / "indices" / "qdrant_legal"


def build_production_qdrant_index(
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    article_embeddings_path: Path = DEFAULT_ARTICLE_EMBEDDINGS,
    cache_path: Path = DEFAULT_CACHE_FILE,
    storage_path: Path = DEFAULT_QDRANT_STORAGE,
    batch_size: int = 500,
) -> int:
    print("=" * 75)
    print("PHASE 8: XÂY DỰNG CHỈ MỤC VECTOR QDRANT PRODUCTION (PERSISTENT EMBEDDED)")
    print("=" * 75)

    if not chunks_path.exists():
        print(f"[FAIL] Không tìm thấy tập chunk đông băng tại: {chunks_path}")
        return 1

    # 1. Nạp Chunk Corpus
    print(f"1. Nạp Phase 7 Frozen Chunks từ: {chunks_path.name}...")
    t0 = time.time()
    chunks_table = pq.read_table(chunks_path)
    chunks_df = chunks_table.to_pandas()
    num_chunks = len(chunks_df)
    print(f"   -> Đã nạp {num_chunks:,} chunks trong {time.time() - t0:.2f}s")

    # 2. Nạp Embeddings
    print("2. Nạp ma trận embedding Phase 6 và bộ nhớ đệm...")
    article_embeddings = np.load(article_embeddings_path)
    corpus_df = pq.read_table(corpus_path, columns=["global_article_id", "text"]).to_pandas()
    art_len_map = {
        int(r["global_article_id"]): len(str(r["text"])) if r["text"] else 0
        for _, r in corpus_df.iterrows()
    }

    cache_dict: Dict[str, np.ndarray] = {}
    if cache_path.exists():
        loaded = np.load(cache_path)
        for k in loaded.files:
            cache_dict[k] = loaded[k]
        print(f"   -> Đã nạp {len(cache_dict):,} vectors từ cache.")

    # 3. Khởi tạo Qdrant Store
    print(f"3. Khởi tạo Qdrant Vector Store tại: {storage_path}...")
    qdrant_store = QdrantVectorStore(
        storage_path=storage_path,
        dimension=1024,
        collection_name="legal_chunks",
    )

    # 4. Chuẩn bị VectorPoints
    print(f"4. Chuẩn bị {num_chunks:,} VectorPoint có định danh tất định (UUID v5)...")
    c_art_ids = chunks_df["global_article_id"].to_numpy(dtype=np.int32)
    c_chunk_ids = chunks_df["chunk_id"].tolist()
    c_law_ids = chunks_df["law_id"].tolist()
    c_aids = chunks_df["article_identifier"].tolist()
    c_orders = chunks_df["chunk_order"].to_numpy(dtype=np.int32)
    c_totals = chunks_df["total_chunks"].to_numpy(dtype=np.int32)
    c_versions = chunks_df["chunking_version"].tolist()
    c_strategies = chunks_df["chunking_strategy"].tolist()
    c_tokens = chunks_df["token_count"].to_numpy(dtype=np.int32)
    c_hashes = chunks_df["content_hash"].tolist()
    c_texts = chunks_df["chunk_text"].tolist()
    c_starts = chunks_df["char_start"].to_numpy(dtype=np.int64)
    c_ends = chunks_df["char_end"].to_numpy(dtype=np.int64)

    points: List[VectorPoint] = []
    missing_vectors = 0

    for i in range(num_chunks):
        art_id = int(c_art_ids[i])
        ch_id = str(c_chunk_ids[i])
        pt_id = generate_deterministic_point_id(ch_id)

        # Lấy vector
        if c_totals[i] == 1 and c_starts[i] == 0 and c_ends[i] == art_len_map.get(art_id, 0):
            vec = article_embeddings[art_id].tolist()
        elif c_hashes[i] in cache_dict:
            vec = cache_dict[c_hashes[i]].tolist()
        else:
            missing_vectors += 1
            continue

        pt = VectorPoint(
            point_id=pt_id,
            chunk_id=ch_id,
            vector=vec,
            global_article_id=art_id,
            law_id=str(c_law_ids[i]),
            article_identifier=str(c_aids[i]),
            chunk_order=int(c_orders[i]),
            total_chunks=int(c_totals[i]),
            chunking_version=str(c_versions[i]),
            token_count=int(c_tokens[i]),
            content_hash=str(c_hashes[i]),
            chunking_strategy=str(c_strategies[i]),
            chunk_text=str(c_texts[i]),
            is_active=True,
            char_start=int(c_starts[i]),
            char_end=int(c_ends[i]),
        )
        points.append(pt)

    if missing_vectors > 0:
        print(f"[FAIL] Có {missing_vectors} chunks chưa có embedding vector!")
        return 1

    print(f"   -> Đã tạo thành công {len(points):,} VectorPoints.")

    # 5. Thực hiện Upsert vào Qdrant
    existing_count = qdrant_store.count()
    if existing_count == num_chunks:
        print(f"   -> Collection đã có sẵn đủ {existing_count:,} points hợp lệ. Bỏ qua bước nạp lại.")
    else:
        print(f"5. Bắt đầu nạp (upsert) vào Qdrant với batch_size={batch_size}...")
        t_upsert = time.time()
        total_upserted = qdrant_store.upsert(points, batch_size=batch_size)
        dur_upsert = time.time() - t_upsert
        upsert_qps = total_upserted / dur_upsert if dur_upsert > 0 else 0
        print(f"   -> Nạp thành công {total_upserted:,} vectors trong {dur_upsert:.2f}s ({upsert_qps:.1f} vectors/s)")

    # 6. Kiểm tra đối soát (Reconciliation)
    print("6. Thực hiện kiểm toán đối soát (Reconciliation Audit)...")
    report = run_reconciliation(
        vector_store=qdrant_store,
        authoritative_chunks_path=chunks_path,
        sample_size=1000,
    )
    print(f"   * Tổng số chunks chuẩn     : {report.total_authoritative_chunks:,}")
    print(f"   * Tổng số points trong store: {report.total_vector_points:,}")
    print(f"   * Số points thiếu (missing) : {report.missing_points_count}")
    print(f"   * Số points cũ lệch (stale) : {report.stale_points_count}")
    print(f"   * Số points mồ côi (orphan) : {report.orphan_points_count}")
    print(f"   * Sai lệch metadata         : {report.metadata_mismatch_count}")
    print(f"   * Toàn vẹn đối soát         : {'PASS' if report.is_consistent else 'FAIL'}")

    assert report.is_consistent, "[FAIL] Kiểm toán đối soát Qdrant không đồng nhất!"

    # 7. Thử nghiệm tìm kiếm truy vấn mẫu
    print("7. Thử nghiệm truy vấn mẫu trên Qdrant...")
    sample_query_vec = np.random.randn(1024).astype(np.float32)
    sample_query_vec /= np.linalg.norm(sample_query_vec)
    search_res = qdrant_store.search(sample_query_vec, limit=5)
    print(f"   -> Trả về {len(search_res)} kết quả. Top 1: chunk_id={search_res[0].chunk_id}, score={search_res[0].score:.4f}")

    print("=" * 75)
    print(f"[PASS] XÂY DỰNG CHỈ MỤC QDRANT VECTOR PRODUCTION THÀNH CÔNG!")
    print(f"       Vị trí lưu trữ: {storage_path}")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xây dựng chỉ mục vector Qdrant production")
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--storage-path", type=Path, default=DEFAULT_QDRANT_STORAGE)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    sys.exit(
        build_production_qdrant_index(
            chunks_path=args.chunks_path,
            storage_path=args.storage_path,
            batch_size=args.batch_size,
        )
    )
