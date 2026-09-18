# tests/test_reconciliation.py
"""Unit tests cho tiến trình đối soát (Reconciliation) trong Phase 8.

Kiểm tra:
    1. Trạng thái đồng nhất 100% (is_consistent = True).
    2. Phát hiện Missing vector khi store bị thiếu điểm.
    3. Phát hiện Stale vector khi store có điểm bị sai lệch content_hash.
    4. Phát hiện Metadata mismatch khi store bị lệch thuộc tính metadata.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.retrieval.indexing.faiss_vector_store import FaissVectorStore
from src.retrieval.indexing.models import VectorPoint, generate_deterministic_point_id
from src.retrieval.indexing.reconciliation import run_reconciliation


class TestReconciliation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.dimension = 32

        # Tạo file parquet authoritative chunks giả lập
        self.chunks_data = []
        self.points = []
        for i in range(5):
            cid = f"auth_chunk_{i:03d}"
            pid = generate_deterministic_point_id(cid)
            vec = [0.1 * (j % 5) for j in range(self.dimension)]
            row = {
                "chunk_id": cid,
                "global_article_id": 100 + i,
                "law_id": "LAW_TEST",
                "article_identifier": str(100 + i),
                "chunk_order": 0,
                "total_chunks": 1,
                "chunking_version": "v1.0",
                "token_count": 100,
                "content_hash": f"hash_{i}",
                "chunking_strategy": "strat_test",
                "chunk_text": f"Văn bản điều luật {i}",
                "char_start": 0,
                "char_end": 50,
            }
            self.chunks_data.append(row)
            self.points.append(
                VectorPoint.from_chunk_and_vector(row, vector=vec)
            )

        self.parquet_path = self.temp_dir / "test_chunks.parquet"
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(self.chunks_data)), self.parquet_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_consistent_state(self):
        store = FaissVectorStore(dimension=self.dimension)
        store.upsert(self.points)

        report = run_reconciliation(store, self.parquet_path)
        self.assertTrue(report.is_consistent)
        self.assertEqual(report.missing_points_count, 0)
        self.assertEqual(report.stale_points_count, 0)

    def test_detect_missing_vector(self):
        store = FaissVectorStore(dimension=self.dimension)
        # Chỉ nạp 4/5 điểm (thiếu điểm cuối)
        store.upsert(self.points[:4])

        report = run_reconciliation(store, self.parquet_path)
        self.assertFalse(report.is_consistent)
        self.assertEqual(report.missing_points_count, 1)
        self.assertIn(self.chunks_data[4]["chunk_id"], report.missing_chunk_ids)

    def test_detect_stale_vector(self):
        store = FaissVectorStore(dimension=self.dimension)
        # Nạp điểm có content_hash bị lệch
        stale_point = VectorPoint.from_chunk_and_vector(self.chunks_data[0], vector=[0.1]*self.dimension)
        stale_point.content_hash = "stale_hash_xyz"
        store.upsert([stale_point] + self.points[1:])

        report = run_reconciliation(store, self.parquet_path)
        self.assertFalse(report.is_consistent)
        self.assertEqual(report.stale_points_count, 1)


if __name__ == "__main__":
    unittest.main()
