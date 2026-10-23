"""Unit tests for ArticleReranker and neural cross-encoder reranking logic."""

import unittest

from src.retrieval.base import RetrievalResult
from src.retrieval.reranking.cross_encoder import ArticleReranker


class TestArticleReranker(unittest.TestCase):
    def setUp(self) -> None:
        def mock_predict(pairs):
            scores = []
            for query, doc in pairs:
                if "chính xác" in doc:
                    scores.append(10.0)
                elif "thứ nhì" in doc:
                    scores.append(5.0)
                elif "thứ ba" in doc:
                    scores.append(2.0)
                else:
                    scores.append(-1.0)
            return scores

        self.reranker = ArticleReranker(
            model_name_or_path="mock-reranker",
            custom_predict_fn=mock_predict,
        )

    def test_reranking_reorders_candidates(self) -> None:
        candidate_ids = [101, 102, 103, 104]
        candidate_texts = [
            "Văn bản không liên quan gì",
            "Văn bản thứ ba liên quan",
            "Văn bản chính xác nhất",
            "Văn bản thứ nhì liên quan",
        ]

        results = self.reranker.rerank(
            query="Tìm kiếm văn bản chính xác",
            candidate_ids=candidate_ids,
            candidate_texts=candidate_texts,
        )

        self.assertEqual(len(results), 4)
        self.assertEqual(results[0].article_id, 103)
        self.assertEqual(results[0].rank, 1)
        self.assertEqual(results[0].score, 10.0)

        self.assertEqual(results[1].article_id, 104)
        self.assertEqual(results[1].rank, 2)

        self.assertEqual(results[2].article_id, 102)
        self.assertEqual(results[2].rank, 3)

        self.assertEqual(results[3].article_id, 101)
        self.assertEqual(results[3].rank, 4)

    def test_candidate_pool_size_truncation(self) -> None:
        candidate_ids = [101, 102, 103, 104, 105]
        candidate_texts = [
            "Văn bản bình thường",
            "Văn bản chính xác",
            "Văn bản thứ nhì",
            "Văn bản chính xác ở xa",
            "Văn bản cuối cùng",
        ]

        results = self.reranker.rerank(
            query="Tìm kiếm",
            candidate_ids=candidate_ids,
            candidate_texts=candidate_texts,
            candidate_pool_size=3,
        )

        self.assertEqual(len(results), 5)
        self.assertEqual(results[0].article_id, 102)
        self.assertEqual(results[0].rank, 1)
        self.assertEqual(results[1].article_id, 103)
        self.assertEqual(results[1].rank, 2)
        self.assertEqual(results[2].article_id, 101)
        self.assertEqual(results[2].rank, 3)

        self.assertEqual(results[3].article_id, 104)
        self.assertEqual(results[3].rank, 4)
        self.assertEqual(results[4].article_id, 105)
        self.assertEqual(results[4].rank, 5)

    def test_top_k_output_filtering(self) -> None:
        candidate_ids = [1, 2, 3, 4, 5]
        candidate_texts = ["Doc 1", "chính xác", "Doc 3", "thứ nhì", "Doc 5"]

        results = self.reranker.rerank(
            query="Test",
            candidate_ids=candidate_ids,
            candidate_texts=candidate_texts,
            top_k=2,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].article_id, 2)
        self.assertEqual(results[1].article_id, 4)

    def test_empty_candidates(self) -> None:
        results = self.reranker.rerank(
            query="Query",
            candidate_ids=[],
            candidate_texts=[],
        )
        self.assertEqual(results, [])

    def test_mismatched_lengths_raises_error(self) -> None:
        with self.assertRaises(ValueError):
            self.reranker.rerank(
                query="Query",
                candidate_ids=[1, 2],
                candidate_texts=["Doc 1"],
            )


if __name__ == "__main__":
    unittest.main()
