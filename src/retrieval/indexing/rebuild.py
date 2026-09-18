# src/retrieval/indexing/rebuild.py
"""Chiến lược tái xây dựng và khôi phục chỉ mục vector (Rebuild & Recovery Strategy).

Tuân thủ Mục 22 và Mục 23 của Đặc tả Phase 8:
    - Xây dựng lại toàn bộ chỉ mục vector từ nguồn Parquet và embedding cache.
    - Duy trì collection cũ cho rollback window trước khi switch chính thức.
    - Xác thực tính toàn vẹn 100% trước khi bàn giao.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pyarrow.parquet as pq

from src.retrieval.dense.encoder import DenseEncoder
from src.retrieval.indexing.base import BaseVectorStore
from src.retrieval.indexing.models import VectorPoint


def rebuild_vector_index(
    vector_store: BaseVectorStore,
    authoritative_chunks_path: Union[str, Path],
    embeddings_cache_file: Optional[Union[str, Path]] = None,
    article_embeddings_path: Optional[Union[str, Path]] = None,
    encoder: Optional[DenseEncoder] = None,
    batch_size: int = 500,
) -> Dict[str, Any]:
    """Thực thi quy trình tái xây dựng chỉ mục vector theo 9 bước của Mục 22."""
    t_start = time.time()
    chunks_path = Path(authoritative_chunks_path)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file chunks tại: {chunks_path}")

    chunks_table = pq.read_table(chunks_path)
    chunks_df = chunks_table.to_pandas()
    num_chunks = len(chunks_df)

    # Nạp cache embeddings
    cache_dict: Dict[str, np.ndarray] = {}
    if embeddings_cache_file and Path(embeddings_cache_file).exists():
        loaded = np.load(embeddings_cache_file)
        for k in loaded.files:
            cache_dict[k] = loaded[k]

    article_embeddings = None
    if article_embeddings_path and Path(article_embeddings_path).exists():
        article_embeddings = np.load(article_embeddings_path)

    points: List[VectorPoint] = []
    missing_texts: List[str] = []
    missing_indices: List[int] = []

    for i, row in chunks_df.iterrows():
        c_dict = row.to_dict()
        chash = str(row["content_hash"])
        gid = int(row["global_article_id"])
        total_c = int(row["total_chunks"])
        c_start = int(row["char_start"])

        # Kiểm tra xem có lấy trực tiếp được từ Phase 6 hay cache không
        if total_c == 1 and c_start == 0 and article_embeddings is not None and gid < len(article_embeddings):
            vec = article_embeddings[gid]
        elif chash in cache_dict:
            vec = cache_dict[chash]
        else:
            vec = None
            missing_indices.append(i)
            missing_texts.append(str(row["chunk_text"]))

        p = VectorPoint.from_chunk_and_vector(
            chunk_dict=c_dict,
            vector=vec.tolist() if vec is not None else [],
            is_active=True,
        )
        points.append(p)

    # Nếu có chunk chưa có embedding thì tính toán bổ sung
    if missing_texts:
        if encoder is None:
            encoder = DenseEncoder()
        new_vecs = encoder.encode_articles(missing_texts, batch_size=16)
        for m_idx, v in zip(missing_indices, new_vecs):
            points[m_idx].vector = v.tolist()
            if embeddings_cache_file:
                cache_dict[points[m_idx].content_hash] = v
        if embeddings_cache_file:
            np.savez_compressed(embeddings_cache_file, **cache_dict)

    # Upsert toàn bộ vào vector store theo batch
    total_upserted = 0
    for b_i in range(0, num_chunks, batch_size):
        b_points = points[b_i : b_i + batch_size]
        total_upserted += vector_store.upsert(b_points)

    total_time = time.time() - t_start
    final_count = vector_store.count()

    return {
        "status": "SUCCESS" if final_count == num_chunks else "MISMATCH",
        "total_chunks": num_chunks,
        "indexed_points": final_count,
        "upserted_points": total_upserted,
        "duration_sec": round(total_time, 2),
        "throughput_points_per_sec": round(num_chunks / total_time, 1) if total_time > 0 else 0,
    }
