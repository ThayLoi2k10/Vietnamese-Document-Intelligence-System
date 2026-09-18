# tests/test_vector_store.py
"""Unit tests cho các thành phần kho vector (FaissVectorStore, QdrantVectorStore) và Retriever (Phase 8)."""

import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.chunked_retriever import ChunkedVectorRetriever
from src.retrieval.indexing.faiss_vector_store import FaissVectorStore
from src.retrieval.indexing.models import VectorPoint, generate_deterministic_point_id
from src.retrieval.indexing.qdrant_store import QdrantVectorStore


class MockEncoder:
    """Mock encoder trả về vector định trước cho test nhanh độc lập."""

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension

    def encode_queries(self, queries, batch_size=32):
        rng = np.random.default_rng(99)
        vecs = rng.standard_normal((len(queries), self.dimension)).astype(np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def encode_articles(self, articles, batch_size=16):
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((len(articles), self.dimension)).astype(np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class TestVectorStores(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.dimension = 64
        rng = np.random.default_rng(42)
        raw_vecs = rng.standard_normal((10, self.dimension)).astype(np.float32)
        norm_vecs = raw_vecs / np.linalg.norm(raw_vecs, axis=1, keepdims=True)

        self.sample_points: list[VectorPoint] = []
        for i in range(10):
            cid = f"chunk_test_{i:03d}"
            pid = generate_deterministic_point_id(cid)
            self.sample_points.append(
                VectorPoint(
                    point_id=pid,
                    chunk_id=cid,
                    vector=norm_vecs[i].tolist(),
                    global_article_id=100 + (i % 5),
                    law_id=f"LAW_{(i % 2):02d}",
                    article_identifier=str(100 + (i % 5)),
                    chunk_order=i % 2,
                    total_chunks=2,
                    chunking_version="test_v1",
                    token_count=128,
                    content_hash=f"hash_{i}",
                    chunking_strategy="test_strategy",
                    chunk_text=f"Nội dung chunk thử nghiệm số {i}",
                    is_active=True,
                )
            )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_faiss_vector_store(self):
        store = FaissVectorStore(dimension=self.dimension)
        store.upsert(self.sample_points)
        self.assertEqual(store.count(), 10)

        # Tìm kiếm tương đồng
        q_vec = np.array(self.sample_points[0].vector, dtype=np.float32)
        results = store.search(q_vec, limit=3)
        self.assertEqual(len(results), 3)
        # Top-1 phải trùng chính xác với điểm 0
        self.assertEqual(results[0].point_id, self.sample_points[0].point_id)
        self.assertAlmostEqual(results[0].score, 1.0, places=4)

        # Lọc metadata (pre-filter theo law_id)
        filtered_results = store.search(q_vec, limit=10, filter_dict={"law_id": "LAW_01"})
        self.assertTrue(len(filtered_results) > 0)
        for r in filtered_results:
            self.assertEqual(r.payload.get("law_id"), "LAW_01")

        # Xóa điểm
        store.delete([self.sample_points[0].point_id])
        self.assertEqual(store.count(), 9)

        # Lưu và nạp lại
        save_path = self.temp_dir / "faiss_save"
        store.save(save_path)
        loaded_store = FaissVectorStore.load(save_path)
        self.assertEqual(loaded_store.count(), 9)

    def test_qdrant_vector_store_lifecycle_and_persistence(self):
        qdrant_dir = self.temp_dir / "qdrant_test_dir"
        store = QdrantVectorStore(storage_path=qdrant_dir, dimension=self.dimension)
        store.upsert(self.sample_points)
        self.assertEqual(store.count(), 10)

        # Idempotency
        store.upsert(self.sample_points)
        self.assertEqual(store.count(), 10)

        # Tìm kiếm
        q_vec = np.array(self.sample_points[0].vector, dtype=np.float32)
        results = store.search(q_vec, limit=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].point_id, self.sample_points[0].point_id)

        # Lọc metadata (pre-filter theo global_article_id)
        target_art = self.sample_points[0].global_article_id
        filtered_results = store.search(q_vec, limit=10, filter_dict={"global_article_id": target_art})
        self.assertTrue(len(filtered_results) > 0)
        for r in filtered_results:
            self.assertEqual(r.payload.get("global_article_id"), target_art)

        # Persistence across restart
        del store
        restarted_store = QdrantVectorStore.load(qdrant_dir)
        self.assertEqual(restarted_store.count(), 10)
        retrieved = restarted_store.get([self.sample_points[0].point_id])[0]
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.chunk_id, self.sample_points[0].chunk_id)

    def test_chunked_retriever_contract(self):
        store = FaissVectorStore(dimension=self.dimension)
        store.upsert(self.sample_points)
        mock_enc = MockEncoder(dimension=self.dimension)

        retriever = ChunkedVectorRetriever(vector_store=store, encoder=mock_enc, chunk_depth=10)
        self.assertIsInstance(retriever, BaseRetriever)
        self.assertTrue(retriever.is_fitted)

        results = retriever.search("Quy định điều luật", top_k=3)
        self.assertEqual(len(results), 3)
        self.assertIsInstance(results[0], RetrievalResult)

        # 1-indexed rank, score giảm dần, không lặp article_id
        seen_ids = set()
        for idx, r in enumerate(results, start=1):
            self.assertEqual(r.rank, idx)
            self.assertNotIn(r.article_id, seen_ids)
            seen_ids.add(r.article_id)
            if idx > 1:
                self.assertGreaterEqual(results[idx - 2].score, r.score)


if __name__ == "__main__":
    unittest.main()
