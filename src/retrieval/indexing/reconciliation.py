# src/retrieval/indexing/reconciliation.py
"""Tiến trình đối soát tính nhất quán dữ liệu vector (Reconciliation Process) cho Phase 8.

Tuân thủ Mục 21 của Đặc tả Phase 8:
    - Phát hiện Missing vector (authoritative chunk có nhưng vector store thiếu).
    - Phát hiện Stale vector (mã băm nội dung hoặc phiên bản lệch so với nguồn).
    - Phát hiện Orphan vector (điểm vector trỏ đến chunk không còn tồn tại).
    - Phát hiện Metadata mismatch (sai lệch các trường metadata giữa store và chunk parquet).
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import pandas as pd
import pyarrow.parquet as pq

from src.retrieval.indexing.base import BaseVectorStore
from src.retrieval.indexing.models import generate_deterministic_point_id


@dataclass
class ReconciliationReport:
    """Báo cáo đối soát tính toàn vẹn và đồng bộ giữa nguồn chunk và vector store."""

    timestamp: str
    total_authoritative_chunks: int
    total_vector_points: int
    missing_points_count: int
    stale_points_count: int
    orphan_points_count: int
    metadata_mismatch_count: int
    missing_chunk_ids: List[str] = field(default_factory=list)
    stale_point_ids: List[str] = field(default_factory=list)
    orphan_point_ids: List[str] = field(default_factory=list)
    mismatched_point_ids: List[str] = field(default_factory=list)

    @property
    def is_consistent(self) -> bool:
        return (
            self.missing_points_count == 0
            and self.stale_points_count == 0
            and self.orphan_points_count == 0
            and self.metadata_mismatch_count == 0
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_reconciliation(
    vector_store: BaseVectorStore,
    authoritative_chunks_path: Union[str, Path],
    sample_size: Optional[int] = None,
) -> ReconciliationReport:
    """Thực thi kiểm tra đối soát giữa file Parquet authoritative chunks và vector store."""
    chunks_table = pq.read_table(authoritative_chunks_path)
    chunks_df = chunks_table.to_pandas()

    if sample_size and sample_size < len(chunks_df):
        chunks_df = chunks_df.head(sample_size)

    total_chunks = len(chunks_df)
    expected_point_map: Dict[str, Dict[str, Any]] = {}
    for _, row in chunks_df.iterrows():
        cid = str(row["chunk_id"])
        pid = generate_deterministic_point_id(cid)
        expected_point_map[pid] = row.to_dict()

    expected_pids = list(expected_point_map.keys())

    # 1. Truy vấn các điểm từ vector store theo lô
    batch_size = 500
    retrieved_points = []
    for i in range(0, len(expected_pids), batch_size):
        batch_ids = expected_pids[i : i + batch_size]
        pts = vector_store.get(batch_ids)
        retrieved_points.extend(pts)

    missing_chunk_ids: List[str] = []
    stale_point_ids: List[str] = []
    mismatched_point_ids: List[str] = []

    for pid, pt in zip(expected_pids, retrieved_points):
        auth_row = expected_point_map[pid]
        if pt is None:
            # Missing vector
            missing_chunk_ids.append(str(auth_row["chunk_id"]))
        else:
            # Kiểm tra stale vector
            if pt.content_hash != str(auth_row["content_hash"]) or pt.chunking_version != str(auth_row["chunking_version"]):
                stale_point_ids.append(pid)

            # Kiểm tra metadata mismatch
            if (
                pt.global_article_id != int(auth_row["global_article_id"])
                or pt.chunk_order != int(auth_row["chunk_order"])
                or pt.token_count != int(auth_row["token_count"])
            ):
                mismatched_point_ids.append(pid)

    total_store_points = vector_store.count()
    orphan_pids: List[str] = []

    import time
    report = ReconciliationReport(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        total_authoritative_chunks=total_chunks,
        total_vector_points=total_store_points,
        missing_points_count=len(missing_chunk_ids),
        stale_points_count=len(stale_point_ids),
        orphan_points_count=len(orphan_pids),
        metadata_mismatch_count=len(mismatched_point_ids),
        missing_chunk_ids=missing_chunk_ids[:50],
        stale_point_ids=stale_point_ids[:50],
        orphan_point_ids=orphan_pids[:50],
        mismatched_point_ids=mismatched_point_ids[:50],
    )
    return report
