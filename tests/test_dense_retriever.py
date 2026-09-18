# tests/test_dense_retriever.py
"""Unit tests cho DenseRetriever, tuân thủ hợp đồng BaseRetriever và RetrievalResult."""

from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.dense.index import DenseIndex
from src.retrieval.dense.retriever import DenseRetriever


class MockDenseEncoder:
    """Mock encoder trả về vector định trước phục vụ unit test nhanh độc lập."""

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension

    def encode_queries(self, queries, batch_size=32):
        n = len(queries)
        rng = np.random.default_rng(99)
        vecs = rng.standard_normal((n, self.dimension)).astype(np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def encode_articles(self, articles, batch_size=4):
        n = len(articles)
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((n, self.dimension)).astype(np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class TestDenseRetriever(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_encoder = MockDenseEncoder(dimension=1024)
        self.corpus_df = pd.DataFrame(
            {
                "global_article_id": [101, 102, 103, 104, 105],
                "text": ["Điều 1", "Điều 2", "Điều 3", "Điều 4", "Điều 5"],
            }
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_subclass_and_unfitted_guard(self):
        retriever = DenseRetriever(encoder=self.mock_encoder)
        self.assertIsInstance(retriever, BaseRetriever)
        self.assertFalse(retriever.is_fitted)

        with self.assertRaises(RuntimeError):
            retriever.search("Truy vấn khi chưa fit")

    def test_fit_and_search_contract(self):
        retriever = DenseRetriever(encoder=self.mock_encoder)
        retriever.fit(self.corpus_df)

        self.assertTrue(retriever.is_fitted)

        results = retriever.search("Quy định điều luật", top_k=3)

        self.assertEqual(len(results), 3)
        self.assertIsInstance(results[0], RetrievalResult)

        # Kiểm tra kiểu dữ liệu và thuộc tính
        for idx, res in enumerate(results, start=1):
            self.assertIsInstance(res.article_id, int)
            self.assertIsInstance(res.score, float)
            self.assertIsInstance(res.rank, int)
            self.assertEqual(res.rank, idx)  # 1-indexed

        # Kiểm tra thứ tự giảm dần của score
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i].score, results[i + 1].score)

        # Kiểm tra không có ID trùng lặp
        ids = [r.article_id for r in results]
        self.assertEqual(len(ids), len(set(ids)))

    def test_batch_search(self):
        retriever = DenseRetriever(encoder=self.mock_encoder)
        retriever.fit(self.corpus_df)

        queries = ["Câu hỏi 1", "Câu hỏi 2"]
        batch_results = retriever.batch_search(queries, top_k=4)

        self.assertEqual(len(batch_results), 2)
        self.assertEqual(len(batch_results[0]), 4)
        self.assertEqual(len(batch_results[1]), 4)

    def test_save_and_load(self):
        retriever = DenseRetriever(encoder=self.mock_encoder)
        retriever.fit(self.corpus_df)

        save_path = self.temp_dir / "retriever_model"
        retriever.save(save_path)

        loaded_retriever = DenseRetriever.load(save_path, encoder=self.mock_encoder)
        self.assertTrue(loaded_retriever.is_fitted)

        res_orig = retriever.search("Test", top_k=3)
        res_loaded = loaded_retriever.search("Test", top_k=3)

        self.assertEqual([r.article_id for r in res_orig], [r.article_id for r in res_loaded])
        for r1, r2 in zip(res_orig, res_loaded):
            self.assertAlmostEqual(r1.score, r2.score, places=5)


if __name__ == "__main__":
    unittest.main()
