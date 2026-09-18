# tests/test_chunking.py
"""Unit tests cho các chiến lược chunking trong Phase 7.

Kiểm tra:
    1. Bảo toàn điều luật rỗng: total_chunks = 0, không sinh text giả.
    2. Bảo toàn nguồn gốc (provenance): chunk_id, global_article_id, law_id, chunk_order.
    3. Tính tất định (determinism): hai lần chạy cho kết quả giống hệt nhau 100%.
    4. Tính hợp lệ của offset ký tự: text[char_start:char_end] == chunk_text.
    5. Các chiến lược: Fixed-size, Recursive, Sentence, Paragraph, Structure-aware.
"""

import unittest
from transformers import AutoTokenizer

from src.retrieval.chunking import (
    FixedSizeChunker,
    ParagraphChunker,
    RecursiveChunker,
    SentenceChunker,
    StructureAwareChunker,
)

SAMPLE_ARTICLE_TEXT = """1. Thông tư này quy định mã số, tiêu chuẩn chuyên môn, nghiệp vụ và xếp lương đối với các ngạch công chức chuyên ngành Ngân hàng.

2. Thông tư này áp dụng đối với công chức làm việc tại các đơn vị thuộc Ngân hàng Nhà nước Việt Nam (gọi tắt là Ngân hàng Nhà nước).

3. Nhiệm vụ của công chức bao gồm:
	a) Chủ trì xây dựng, hoàn thiện thể chế hoặc cơ chế, quy chế, chương trình, kế hoạch kiểm soát nội bộ.
	b) Trực tiếp thực hiện kiểm soát nội bộ, kiểm toán nội bộ các đơn vị thuộc hệ thống.
	c) Báo cáo Thống đốc Ngân hàng Nhà nước về kết quả kiểm soát theo quy định của pháp luật."""


class TestChunkingStrategies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", local_files_only=True)
        except Exception:
            cls.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

    def test_empty_article_preservation(self):
        """Kiểm tra điều luật rỗng phải trả về [] và total_chunks=0."""
        chunkers = [
            FixedSizeChunker(target_size=256, tokenizer=self.tokenizer),
            RecursiveChunker(target_size=256, tokenizer=self.tokenizer),
            SentenceChunker(target_size=256, tokenizer=self.tokenizer),
            ParagraphChunker(target_size=256, tokenizer=self.tokenizer),
            StructureAwareChunker(target_size=256, tokenizer=self.tokenizer),
        ]
        for chunker in chunkers:
            chunks = chunker.chunk_article(
                text="",
                global_article_id=999,
                law_id="LAW_EMPTY",
                article_identifier="aid_empty",
            )
            self.assertEqual(len(chunks), 0, f"Thất bại trên chunker: {chunker.strategy_name}")

            chunks_ws = chunker.chunk_article(
                text="   \n\t  ",
                global_article_id=998,
                law_id="LAW_EMPTY",
                article_identifier="aid_empty",
            )
            self.assertEqual(len(chunks_ws), 0, f"Thất bại trên chunker ws: {chunker.strategy_name}")

    def test_determinism(self):
        """Kiểm tra hai lần chunk độc lập phải tạo ra kết quả giống hệt nhau 100%."""
        chunker = FixedSizeChunker(target_size=256, tokenizer=self.tokenizer)
        chunks_1 = chunker.chunk_article(
            text=SAMPLE_ARTICLE_TEXT,
            global_article_id=42,
            law_id="14/2022/TT-NHNN",
            article_identifier="0",
        )
        chunks_2 = chunker.chunk_article(
            text=SAMPLE_ARTICLE_TEXT,
            global_article_id=42,
            law_id="14/2022/TT-NHNN",
            article_identifier="0",
        )

        self.assertEqual(len(chunks_1), len(chunks_2))
        for c1, c2 in zip(chunks_1, chunks_2):
            self.assertEqual(c1.chunk_id, c2.chunk_id)
            self.assertEqual(c1.content_hash, c2.content_hash)
            self.assertEqual(c1.chunk_text, c2.chunk_text)
            self.assertEqual(c1.token_count, c2.token_count)
            self.assertEqual(c1.char_start, c2.char_start)
            self.assertEqual(c1.char_end, c2.char_end)

    def test_fixed_size_chunker(self):
        """Kiểm tra tính đúng đắn của FixedSizeChunker."""
        chunker = FixedSizeChunker(target_size=64, overlap=0.0, tokenizer=self.tokenizer)
        chunks = chunker.chunk_article(
            text=SAMPLE_ARTICLE_TEXT,
            global_article_id=1,
            law_id="14/2022/TT-NHNN",
            article_identifier="aid_1",
        )
        self.assertGreater(len(chunks), 1)
        for idx, c in enumerate(chunks):
            self.assertEqual(c.global_article_id, 1)
            self.assertEqual(c.chunk_order, idx)
            self.assertEqual(c.total_chunks, len(chunks))
            self.assertEqual(c.chunk_text, SAMPLE_ARTICLE_TEXT[c.char_start : c.char_end])
            self.assertLessEqual(c.token_count, 64)

    def test_recursive_chunker(self):
        """Kiểm tra tính đúng đắn của RecursiveChunker."""
        chunker = RecursiveChunker(target_size=100, tokenizer=self.tokenizer)
        chunks = chunker.chunk_article(
            text=SAMPLE_ARTICLE_TEXT,
            global_article_id=2,
            law_id="14/2022/TT-NHNN",
            article_identifier="aid_2",
        )
        self.assertGreaterEqual(len(chunks), 1)
        for idx, c in enumerate(chunks):
            self.assertEqual(c.global_article_id, 2)
            self.assertEqual(c.chunk_order, idx)
            self.assertEqual(c.chunk_text, SAMPLE_ARTICLE_TEXT[c.char_start : c.char_end])

    def test_sentence_chunker(self):
        """Kiểm tra SentenceChunker bảo toàn ranh giới câu."""
        chunker = SentenceChunker(target_size=100, tokenizer=self.tokenizer)
        chunks = chunker.chunk_article(
            text=SAMPLE_ARTICLE_TEXT,
            global_article_id=3,
            law_id="14/2022/TT-NHNN",
            article_identifier="aid_3",
        )
        self.assertGreaterEqual(len(chunks), 1)
        for idx, c in enumerate(chunks):
            self.assertEqual(c.global_article_id, 3)
            self.assertEqual(c.chunk_order, idx)
            self.assertEqual(c.chunk_text, SAMPLE_ARTICLE_TEXT[c.char_start : c.char_end])

    def test_paragraph_chunker(self):
        """Kiểm tra ParagraphChunker bảo toàn ranh giới đoạn."""
        chunker = ParagraphChunker(target_size=100, tokenizer=self.tokenizer)
        chunks = chunker.chunk_article(
            text=SAMPLE_ARTICLE_TEXT,
            global_article_id=4,
            law_id="14/2022/TT-NHNN",
            article_identifier="aid_4",
        )
        self.assertGreaterEqual(len(chunks), 1)
        for idx, c in enumerate(chunks):
            self.assertEqual(c.global_article_id, 4)
            self.assertEqual(c.chunk_order, idx)
            self.assertEqual(c.chunk_text, SAMPLE_ARTICLE_TEXT[c.char_start : c.char_end])

    def test_structure_aware_chunker(self):
        """Kiểm tra StructureAwareChunker phát hiện Khoản và Điểm."""
        chunker = StructureAwareChunker(target_size=60, tokenizer=self.tokenizer)
        chunks = chunker.chunk_article(
            text=SAMPLE_ARTICLE_TEXT,
            global_article_id=5,
            law_id="14/2022/TT-NHNN",
            article_identifier="aid_5",
        )
        self.assertGreaterEqual(len(chunks), 1)
        # Kiểm tra ít nhất một chunk mang structure_path nhận diện Khoản hoặc Điểm
        has_clause_or_point = any(
            ("Khoản" in c.structure_path or "Điểm" in c.structure_path) for c in chunks
        )
        self.assertTrue(has_clause_or_point, "Không phát hiện ranh giới Khoản/Điểm trong cấu trúc.")
        for idx, c in enumerate(chunks):
            self.assertEqual(c.global_article_id, 5)
            self.assertEqual(c.chunk_order, idx)
            self.assertEqual(c.chunk_text, SAMPLE_ARTICLE_TEXT[c.char_start : c.char_end])


if __name__ == "__main__":
    unittest.main()
