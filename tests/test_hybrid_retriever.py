"""Unit tests for HybridRetriever implementation and ranking fusion logic."""

import json
from pathlib import Path
import tempfile
from typing import List
import unittest

from src.retrieval.base import BaseRetriever, RetrievalResult
from src.retrieval.hybrid.retriever import HybridRetriever


class MockRetriever(BaseRetriever):
    def __init__(self, predefined_results: List[RetrievalResult]) -> None:
        super().__init__()
        self.predefined_results = predefined_results
        self.is_fitted = True

    def fit(self, corpus):
        self.is_fitted = True
        return self

    def search(self, query: str, top_k: int = 100) -> List[RetrievalResult]:
        return self.predefined_results[:top_k]

    def batch_search(self, queries: List[str], top_k: int = 100) -> List[List[RetrievalResult]]:
        return [self.predefined_results[:top_k] for _ in queries]

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        return cls([])


class TestHybridRetriever(unittest.TestCase):
    def setUp(self) -> None:
        self.dense_results = [
            RetrievalResult(article_id=101, score=0.9, rank=1),
            RetrievalResult(article_id=102, score=0.8, rank=2),
            RetrievalResult(article_id=103, score=0.7, rank=3),
        ]
        self.bm25_results = [
            RetrievalResult(article_id=102, score=15.0, rank=1),
            RetrievalResult(article_id=104, score=12.0, rank=2),
            RetrievalResult(article_id=101, score=10.0, rank=3),
        ]
        self.mock_dense = MockRetriever(self.dense_results)
        self.mock_bm25 = MockRetriever(self.bm25_results)

    def test_subclasses_base_retriever(self) -> None:
        hybrid = HybridRetriever(self.mock_dense, self.mock_bm25)
        self.assertIsInstance(hybrid, BaseRetriever)
        self.assertTrue(hybrid.is_fitted)

    def test_standard_rrf_scoring(self) -> None:
        k = 10
        hybrid = HybridRetriever(
            self.mock_dense,
            self.mock_bm25,
            fusion_method="rrf",
            k_rrf=k,
        )
        res = hybrid.search("query test", top_k=10)
        self.assertEqual(len(res), 4)

        ranked_ids = [r.article_id for r in res]
        self.assertEqual(ranked_ids, [102, 101, 104, 103])
        self.assertEqual([r.rank for r in res], [1, 2, 3, 4])
        self.assertAlmostEqual(res[0].score, 0.5 / 12 + 0.5 / 11, places=5)
        self.assertAlmostEqual(res[1].score, 0.5 / 11 + 0.5 / 13, places=5)

    def test_weighted_rrf_scoring(self) -> None:
        k = 10
        hybrid = HybridRetriever(
            self.mock_dense,
            self.mock_bm25,
            fusion_method="weighted_rrf",
            k_rrf=k,
            w_dense=0.8,
            w_bm25=0.2,
        )
        res = hybrid.search("query test", top_k=10)
        self.assertEqual(res[0].article_id, 101)
        self.assertEqual(res[1].article_id, 102)

    def test_linear_fusion(self) -> None:
        hybrid = HybridRetriever(
            self.mock_dense,
            self.mock_bm25,
            fusion_method="linear",
            w_dense=0.6,
        )
        res = hybrid.search("query test", top_k=10)
        self.assertEqual(len(res), 4)
        self.assertEqual([r.rank for r in res], [1, 2, 3, 4])

    def test_batch_search(self) -> None:
        hybrid = HybridRetriever(self.mock_dense, self.mock_bm25)
        batch_res = hybrid.batch_search(["q1", "q2", "q3"], top_k=2)
        self.assertEqual(len(batch_res), 3)
        self.assertEqual(len(batch_res[0]), 2)
        self.assertEqual(len(batch_res[1]), 2)
        self.assertEqual(len(batch_res[2]), 2)

    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            HybridRetriever(self.mock_dense, self.mock_bm25, fusion_method="invalid_method")

        with self.assertRaises(ValueError):
            HybridRetriever(self.mock_dense, self.mock_bm25, k_rrf=0)

        with self.assertRaises(ValueError):
            HybridRetriever(self.mock_dense, self.mock_bm25, w_dense=-0.1)

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "hybrid_config.json"
            hybrid = HybridRetriever(
                self.mock_dense,
                self.mock_bm25,
                fusion_method="weighted_rrf",
                k_rrf=15,
                w_dense=0.75,
                w_bm25=0.25,
                candidate_depth=150,
            )
            hybrid.save(save_path)
            self.assertTrue(save_path.exists())

            loaded = HybridRetriever.load(
                save_path,
                dense_retriever=self.mock_dense,
                bm25_retriever=self.mock_bm25,
            )
            self.assertEqual(loaded.fusion_method, "weighted_rrf")
            self.assertEqual(loaded.k_rrf, 15)
            self.assertEqual(loaded.w_dense, 0.75)
            self.assertEqual(loaded.w_bm25, 0.25)
            self.assertEqual(loaded.candidate_depth, 150)


if __name__ == "__main__":
    unittest.main()
