# tests/unit/context/test_context.py
"""Unit tests cho module Context Assembly Phase 9.

Tuân thủ Mục 22.2: Context unit tests:
    - Top-k selection & Score thresholding
    - Duplicate chunk removal & Same-article policy
    - Relevance ordering vs Document/legal ordering
    - Token budget enforcement & Deterministic overflow behavior
"""

import unittest

from src.rag.context.context_assembler import ContextAssembler
from src.rag.context.domain.token_counter import CharacterTokenCounter
from src.rag.context.policies.policy_types import (
    BudgetOverflowPolicy,
    ContextPolicy,
    DeduplicationPolicy,
    OrderingStrategy,
)
from src.rag.errors.exceptions import ContextBudgetExceededError
from src.rag.retrieval.domain.models import RetrievalItem
from tests.fixtures.sample_retrieval import make_sample_retrieval_items


class TestContextAssemblyModule(unittest.TestCase):
    """Kiểm tra logic lắp ráp, khử trùng lặp và ngân sách ngữ cảnh."""

    def test_top_k_selection_and_score_thresholding(self) -> None:
        """Kiểm tra chọn top-k và lọc theo ngưỡng điểm."""
        items = make_sample_retrieval_items()  # scores: 0.88, 0.75, 0.65
        assembler = ContextAssembler()

        # Lọc ngưỡng điểm >= 0.70 và top_k = 1
        policy = ContextPolicy(context_top_k=1, score_threshold=0.70)
        ctx = assembler.assemble(items, policy=policy, query="test")

        self.assertEqual(len(ctx.items), 1)
        self.assertEqual(ctx.items[0].source_score, 0.88)
        self.assertEqual(ctx.items[0].context_item_id, "[1]")

    def test_deduplication_exact_chunk_id(self) -> None:
        """Kiểm tra loại bỏ chunk trùng lặp ID."""
        items = [
            RetrievalItem(
                chunk_id="chk_dup",
                law_id="ND-01",
                text="Văn bản A",
                rank=1,
                score=0.9,
            ),
            RetrievalItem(
                chunk_id="chk_dup",  # Trùng lặp
                law_id="ND-01",
                text="Văn bản A lặp lại",
                rank=2,
                score=0.8,
            ),
            RetrievalItem(
                chunk_id="chk_unique",
                law_id="ND-01",
                text="Văn bản B",
                rank=3,
                score=0.7,
            ),
        ]
        assembler = ContextAssembler()
        policy = ContextPolicy(
            context_top_k=5,
            deduplication_policy=DeduplicationPolicy.EXACT_CHUNK_ID,
        )
        ctx = assembler.assemble(items, policy=policy, query="test")

        self.assertEqual(len(ctx.items), 2)
        self.assertEqual(ctx.items[0].chunk_id, "chk_dup")
        self.assertEqual(ctx.items[1].chunk_id, "chk_unique")

    def test_deduplication_same_article_limit(self) -> None:
        """Kiểm tra giới hạn số chunk tối đa cho một điều luật."""
        items = [
            RetrievalItem(
                chunk_id="c1",
                global_article_id=10,
                law_id="L1",
                text="Khoản 1",
                rank=1,
                score=0.9,
            ),
            RetrievalItem(
                chunk_id="c2",
                global_article_id=10,
                law_id="L1",
                text="Khoản 2",
                rank=2,
                score=0.8,
            ),
            RetrievalItem(
                chunk_id="c3",
                global_article_id=10,
                law_id="L1",
                text="Khoản 3",
                rank=3,
                score=0.7,
            ),
            RetrievalItem(
                chunk_id="c4",
                global_article_id=20,
                law_id="L1",
                text="Điều khác",
                rank=4,
                score=0.6,
            ),
        ]
        assembler = ContextAssembler()
        policy = ContextPolicy(
            context_top_k=5,
            deduplication_policy=DeduplicationPolicy.SAME_ARTICLE_LIMIT,
            deduplication_options={"max_chunks_per_article": 2},
        )
        ctx = assembler.assemble(items, policy=policy, query="test")

        # Điều 10 chỉ được lấy tối đa 2 chunk, kèm 1 chunk của Điều 20 -> tổng 3
        self.assertEqual(len(ctx.items), 3)
        article_10_items = [
            it for it in ctx.items if it.global_article_id == 10
        ]
        self.assertEqual(len(article_10_items), 2)

    def test_ordering_strategies(self) -> None:
        """Kiểm tra sắp xếp theo điểm số vs theo trật tự văn bản pháp luật."""
        items = [
            RetrievalItem(
                chunk_id="c2",
                law_id="ND-100",
                article_identifier="Điều 2",
                text="Nội dung điều 2",
                rank=1,
                score=0.95,
                chunk_order=1,
            ),
            RetrievalItem(
                chunk_id="c1",
                law_id="ND-100",
                article_identifier="Điều 1",
                text="Nội dung điều 1",
                rank=2,
                score=0.70,
                chunk_order=1,
            ),
        ]
        assembler = ContextAssembler()

        # 1. Theo điểm số giảm dần
        p_score = ContextPolicy(
            ordering_strategy=OrderingStrategy.SCORE_DESCENDING
        )
        ctx_score = assembler.assemble(items, policy=p_score, query="test")
        self.assertEqual(ctx_score.items[0].article_identifier, "Điều 2")

        # 2. Theo thứ tự pháp lý (Điều 1 trước Điều 2)
        p_legal = ContextPolicy(ordering_strategy=OrderingStrategy.LEGAL_ORDER)
        ctx_legal = assembler.assemble(items, policy=p_legal, query="test")
        self.assertEqual(ctx_legal.items[0].article_identifier, "Điều 1")

    def test_token_budget_overflow_raise_error(self) -> None:
        """Kiểm tra ngân sách token bị vượt quá với chính sách RAISE_CONTEXT_BUDGET_ERROR."""
        items = [
            RetrievalItem(
                chunk_id="c1",
                law_id="L1",
                text="Một đoạn văn bản dài có rất nhiều token vượt quá ngân sách...",
                rank=1,
                token_count=100,
            )
        ]
        assembler = ContextAssembler()
        policy = ContextPolicy(
            max_context_tokens=50,
            budget_overflow_policy=BudgetOverflowPolicy.RAISE_CONTEXT_BUDGET_ERROR,
        )
        with self.assertRaises(ContextBudgetExceededError):
            assembler.assemble(items, policy=policy, query="test")

    def test_token_budget_overflow_drop_lowest_priority(self) -> None:
        """Kiểm tra vượt ngân sách với chính sách DROP_LOWEST_PRIORITY_ITEMS."""
        items = [
            RetrievalItem(
                chunk_id="c1",
                law_id="L1",
                text="Đoạn 1",
                rank=1,
                score=0.9,
                token_count=30,
            ),
            RetrievalItem(
                chunk_id="c2",
                law_id="L1",
                text="Đoạn 2",
                rank=2,
                score=0.8,
                token_count=30,
            ),
        ]
        assembler = ContextAssembler()
        # Budget là 40: chỉ vừa Đoạn 1 (30), Đoạn 2 (30) sẽ bị bỏ qua
        policy = ContextPolicy(
            max_context_tokens=40,
            budget_overflow_policy=BudgetOverflowPolicy.DROP_LOWEST_PRIORITY_ITEMS,
        )
        ctx = assembler.assemble(items, policy=policy, query="test")
        self.assertEqual(len(ctx.items), 1)
        self.assertEqual(ctx.items[0].chunk_id, "c1")


if __name__ == "__main__":
    unittest.main()
