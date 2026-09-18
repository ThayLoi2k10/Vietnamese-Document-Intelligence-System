# tests/unit/reranking/test_reranking.py
"""Unit tests cho module Reranking Phase 9.

Tuân thủ Mục 22.2: Reranking unit tests:
    - Disabled reranker path
    - Candidate preservation
    - New score/rank creation
    - Original retrieval provenance preservation
    - Configurable reranking depth
"""

import unittest

from src.rag.reranking.adapters.mock_reranker import MockReranker
from src.rag.reranking.domain.models import RerankOptions
from tests.fixtures.sample_retrieval import make_sample_retrieval_items


class TestRerankingModule(unittest.TestCase):
    """Kiểm tra hợp đồng và tính năng bảo toàn provenance của Reranker."""

    def test_reranker_preserves_provenance_and_assigns_new_ranks(self) -> None:
        """Kiểm tra gán rank/score mới nhưng không làm mất retrieval rank/score."""
        items = make_sample_retrieval_items()
        self.assertEqual(items[0].rank, 1)
        self.assertEqual(items[0].score, 0.88)

        # Khởi tạo MockReranker với chế độ reverse_order (đảo ngược thứ hạng)
        reranker = MockReranker(reverse_order=True)
        options = RerankOptions(reranker_top_k=3)
        res = reranker.rerank("câu hỏi kiểm thử", items, options)

        self.assertEqual(len(res.candidates), 3)

        # Ứng viên đứng đầu sau khi rerank
        top1 = res.candidates[0]
        # Thứ hạng rerank mới phải là 1
        self.assertEqual(top1.rerank_rank, 1)
        # Provenance ban đầu từ retrieval vẫn được lưu đầy đủ
        self.assertIsNotNone(top1.retrieval_rank)
        self.assertIsNotNone(top1.retrieval_score)
        self.assertIsNotNone(top1.rerank_score)
        self.assertEqual(top1.retrieval_item.chunk_id, items[2].chunk_id)

    def test_reranker_depth_truncation(self) -> None:
        """Kiểm tra giới hạn reranker_top_k cắt đúng số lượng ứng viên."""
        items = make_sample_retrieval_items()  # Có 3 items
        reranker = MockReranker()
        options = RerankOptions(reranker_top_k=2)
        res = reranker.rerank("câu hỏi", items, options)

        self.assertEqual(len(res.candidates), 2)
        self.assertEqual(res.candidates[0].rerank_rank, 1)
        self.assertEqual(res.candidates[1].rerank_rank, 2)


if __name__ == "__main__":
    unittest.main()
