# tests/test_retrieval_metrics.py
import math
import unittest
from src.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    compute_ap_at_k,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    evaluate_query_retrieval,
)


class TestRetrievalMetrics(unittest.TestCase):
    def setUp(self):
        # Case 1: Multi-label. Relevant: {10, 20} (2 tài liệu đúng)
        # Retrieved: [10, 99, 20, 88, 77]
        # Rank 1: Doc 10 (Relevant)
        # Rank 2: Doc 99 (Non-relevant)
        # Rank 3: Doc 20 (Relevant)
        self.retrieved_case1 = [10, 99, 20, 88, 77]
        self.relevant_case1 = [10, 20]

    def test_precision_at_k(self):
        self.assertAlmostEqual(
            compute_precision_at_k(self.retrieved_case1, self.relevant_case1, k=1), 1.0
        )
        self.assertAlmostEqual(
            compute_precision_at_k(self.retrieved_case1, self.relevant_case1, k=2), 0.5
        )
        self.assertAlmostEqual(
            compute_precision_at_k(self.retrieved_case1, self.relevant_case1, k=5), 2 / 5
        )

    def test_recall_at_k(self):
        self.assertAlmostEqual(
            compute_recall_at_k(self.retrieved_case1, self.relevant_case1, k=1), 0.5
        )
        self.assertAlmostEqual(
            compute_recall_at_k(self.retrieved_case1, self.relevant_case1, k=2), 0.5
        )
        self.assertAlmostEqual(
            compute_recall_at_k(self.retrieved_case1, self.relevant_case1, k=3), 1.0
        )

    def test_mrr_at_k(self):
        # First relevant doc is at rank 1 -> RR = 1.0
        self.assertAlmostEqual(
            compute_mrr_at_k(self.retrieved_case1, self.relevant_case1, k=10), 1.0
        )
        # Nếu relevant là doc 20 (ở rank 3) -> RR = 1/3
        self.assertAlmostEqual(
            compute_mrr_at_k(self.retrieved_case1, [20], k=10), 1 / 3
        )
        # Nếu không tìm thấy trong top 2 -> RR@2 = 0.0
        self.assertAlmostEqual(
            compute_mrr_at_k(self.retrieved_case1, [20], k=2), 0.0
        )

    def test_ap_at_k(self):
        # Case 1: P@1 = 1/1, P@2 = 1/2 (nhưng không hit), P@3 = 2/3 (hit).
        # Sum precisions = (1.0 + 2/3). Normalizer = min(|R|, 3) = 2.
        # AP = (1.0 + 2/3) / 2 = 5/6 = ~0.8333
        expected_ap = (1.0 + 2 / 3) / 2
        self.assertAlmostEqual(
            compute_ap_at_k(self.retrieved_case1, self.relevant_case1, k=3), expected_ap
        )

    def test_ndcg_at_k(self):
        # DCG@3 = 1/log2(1+1) + 1/log2(3+1) = 1/1 + 1/2 = 1.5
        # IDCG@3 = 1/log2(1+1) + 1/log2(2+1) = 1.0 + 1/1.58496 = ~1.6309
        dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        expected_ndcg = dcg / idcg
        self.assertAlmostEqual(
            compute_ndcg_at_k(self.retrieved_case1, self.relevant_case1, k=3),
            expected_ndcg,
        )

    def test_edge_cases_k_and_empty(self):
        """Kiểm tra các trường hợp biên: k <= 0, tập rỗng, k lớn hơn số kết quả."""
        # k <= 0
        self.assertEqual(compute_precision_at_k(self.retrieved_case1, self.relevant_case1, k=0), 0.0)
        self.assertEqual(compute_recall_at_k(self.retrieved_case1, self.relevant_case1, k=-1), 0.0)
        self.assertEqual(compute_mrr_at_k(self.retrieved_case1, self.relevant_case1, k=0), 0.0)
        self.assertEqual(compute_ap_at_k(self.retrieved_case1, self.relevant_case1, k=-5), 0.0)
        self.assertEqual(compute_ndcg_at_k(self.retrieved_case1, self.relevant_case1, k=0), 0.0)

        # relevant_ids rỗng
        self.assertEqual(compute_recall_at_k(self.retrieved_case1, [], k=5), 0.0)
        self.assertEqual(compute_precision_at_k(self.retrieved_case1, [], k=5), 0.0)
        self.assertEqual(compute_ap_at_k(self.retrieved_case1, [], k=5), 0.0)
        self.assertEqual(compute_ndcg_at_k(self.retrieved_case1, [], k=5), 0.0)

        # retrieved_ids rỗng
        self.assertEqual(compute_precision_at_k([], self.relevant_case1, k=5), 0.0)
        self.assertEqual(compute_recall_at_k([], self.relevant_case1, k=5), 0.0)
        self.assertEqual(compute_mrr_at_k([], self.relevant_case1, k=5), 0.0)

        # k lớn hơn độ dài retrieved_ids
        self.assertAlmostEqual(
            compute_recall_at_k(self.retrieved_case1, self.relevant_case1, k=100), 1.0
        )

    def test_deduplication_protection(self):
        """Đảm bảo không bị lặp hit khi cùng một doc_id xuất hiện nhiều lần."""
        retrieved_duplicates = [10, 10, 99, 10]
        # P@2: chỉ tính 1 hit duy nhất cho doc 10 -> 1 / 2 = 0.5 (thay vì 2 / 2 = 1.0)
        self.assertAlmostEqual(
            compute_precision_at_k(retrieved_duplicates, [10], k=2), 0.5
        )
        # R@4: chỉ tính 1 hit cho doc 10 -> 1 / 1 = 1.0
        self.assertAlmostEqual(
            compute_recall_at_k(retrieved_duplicates, [10], k=4), 1.0
        )

    def test_evaluate_and_aggregate_metrics(self):
        """Kiểm tra pipeline đánh giá query đơn và gộp macro-average."""
        q1_metrics = evaluate_query_retrieval(
            retrieved_ids=[1, 2], relevant_ids=[1], k_list=(1, 2)
        )
        self.assertEqual(q1_metrics["P@1"], 1.0)
        self.assertEqual(q1_metrics["R@1"], 1.0)
        self.assertEqual(q1_metrics["MRR@10"], 1.0)

        q2_metrics = evaluate_query_retrieval(
            retrieved_ids=[9, 8], relevant_ids=[1], k_list=(1, 2)
        )
        self.assertEqual(q2_metrics["P@1"], 0.0)
        self.assertEqual(q2_metrics["R@1"], 0.0)
        self.assertEqual(q2_metrics["MRR@10"], 0.0)

        # Aggregate macro: trung bình của [1.0, 0.0] là 0.5
        agg = aggregate_retrieval_metrics([q1_metrics, q2_metrics])
        self.assertAlmostEqual(agg["P@1"], 0.5)
        self.assertAlmostEqual(agg["R@1"], 0.5)
        self.assertAlmostEqual(agg["MRR@10"], 0.5)

        # Edge case aggregate rỗng
        self.assertEqual(aggregate_retrieval_metrics([]), {})

        # Test dictionary có khóa không đầy đủ
        agg_partial = aggregate_retrieval_metrics([{"P@1": 1.0, "R@1": 0.5}, {"P@1": 0.5}])
        self.assertAlmostEqual(agg_partial["P@1"], 0.75)
        self.assertAlmostEqual(agg_partial["R@1"], 0.25)


if __name__ == "__main__":
    unittest.main()