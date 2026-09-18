# tests/unit/retrieval/test_retrieval.py
"""Unit tests cho module Retrieval Phase 9.

Tuân thủ Mục 22.2: Retrieval unit tests:
    - Normalization of valid retrieval responses
    - Validation of required metadata
    - Rank/score preservation
    - Chunk/article mode handling
    - Empty retrieval behavior
"""

import unittest

from src.rag.errors.exceptions import (
    InvalidRetrievalResultError,
    MissingRequiredMetadataError,
)
from src.rag.retrieval.adapters.mock_retriever import MockRetriever
from src.rag.retrieval.adapters.phase8_adapter import Phase8RetrievalAdapter
from src.rag.retrieval.domain.models import (
    RetrievalItem,
    RetrievalMode,
    RetrievalOptions,
)
from tests.fixtures.sample_retrieval import make_sample_retrieval_items


class TestRetrievalModule(unittest.TestCase):
    """Kiểm tra các hợp đồng và logic của domain Retrieval."""

    def test_retrieval_item_validation_chunk_mode(self) -> None:
        """Kiểm tra xác thực RetrievalItem ở chế độ CHUNK."""
        # Hợp lệ
        item = RetrievalItem(
            chunk_id="chk_1",
            law_id="100/2015",
            text="Văn bản điều luật...",
            rank=1,
            score=0.9,
        )
        item.validate(RetrievalMode.CHUNK)

        # Thiếu chunk_id ở CHUNK mode -> Lỗi
        invalid_item = RetrievalItem(
            chunk_id=None,
            law_id="100/2015",
            text="Văn bản điều luật...",
            rank=1,
        )
        with self.assertRaises(MissingRequiredMetadataError):
            invalid_item.validate(RetrievalMode.CHUNK)

    def test_retrieval_item_validation_article_mode(self) -> None:
        """Kiểm tra xác thực RetrievalItem ở chế độ ARTICLE."""
        # Hợp lệ với global_article_id
        item = RetrievalItem(
            global_article_id=42,
            law_id="100/2015",
            text="Nội dung điều 42...",
            rank=1,
        )
        item.validate(RetrievalMode.ARTICLE)

        # Thiếu cả global_article_id lẫn article_identifier -> Lỗi
        invalid_item = RetrievalItem(
            global_article_id=None,
            article_identifier=None,
            law_id="100/2015",
            text="Nội dung...",
            rank=1,
        )
        with self.assertRaises(MissingRequiredMetadataError):
            invalid_item.validate(RetrievalMode.ARTICLE)

    def test_retrieval_item_invalid_rank(self) -> None:
        """Kiểm tra thứ hạng (rank) phải >= 1."""
        item = RetrievalItem(
            chunk_id="chk_1",
            law_id="100/2015",
            text="Nội dung...",
            rank=0,  # Rank không hợp lệ
        )
        with self.assertRaises(InvalidRetrievalResultError):
            item.validate(RetrievalMode.CHUNK)

    def test_empty_retrieval_is_valid_domain_state(self) -> None:
        """Kiểm tra kết quả truy xuất rỗng là trạng thái hợp lệ, không gây ngoại lệ (Mục 8.4)."""
        retriever = MockRetriever(canned_items=[])
        options = RetrievalOptions(retrieval_top_k=10)
        res = retriever.retrieve("Câu hỏi về vấn đề chưa có luật?", options)

        self.assertTrue(res.is_empty)
        self.assertEqual(len(res.items), 0)
        self.assertEqual(res.query, "Câu hỏi về vấn đề chưa có luật?")

    def test_phase8_adapter_normalization_and_provenance(self) -> None:
        """Kiểm tra Phase8RetrievalAdapter chuẩn hóa đúng và bảo toàn provenance."""
        raw_phase8_records = [
            {
                "chunk_id": "c_101",
                "global_article_id": 505,
                "score": 0.876,
                "rank": 1,
                "payload": {
                    "law_id": "ND-100",
                    "article_identifier": "Điều 10",
                    "chunk_text": "Quy định về bảo trì công trình.",
                    "chunk_order": 1,
                    "total_chunks": 3,
                    "token_count": 30,
                    "custom_db_field": "test_provenance",
                },
            }
        ]

        def fake_search(query: str, top_k: int):
            return raw_phase8_records[:top_k]

        adapter = Phase8RetrievalAdapter(search_fn=fake_search)
        res = adapter.retrieve("quy định bảo trì", RetrievalOptions(retrieval_top_k=5))

        self.assertEqual(len(res.items), 1)
        item = res.items[0]
        self.assertEqual(item.chunk_id, "c_101")
        self.assertEqual(item.global_article_id, 505)
        self.assertEqual(item.law_id, "ND-100")
        self.assertEqual(item.article_identifier, "Điều 10")
        self.assertEqual(item.score, 0.876)
        self.assertEqual(item.rank, 1)
        self.assertEqual(item.source_metadata["custom_db_field"], "test_provenance")


if __name__ == "__main__":
    unittest.main()
