# tests/unit/prompting/test_prompting.py
"""Unit tests cho module Prompt Architecture Phase 9.

Tuân thủ Mục 22.2: Prompting unit tests:
    - Template version selection
    - Deterministic prompt construction
    - Evidence mapping
    - Prompt input validation
    - LLM-visible metadata filtering (loại bỏ vector score/internal rank)
"""

import unittest

from src.rag.context.domain.models import AssembledContext
from src.rag.errors.exceptions import InvalidPromptError
from src.rag.prompting.builders.prompt_builder import PromptBuilder
from src.rag.prompting.templates.registry import PromptTemplateRegistry
from tests.fixtures.sample_retrieval import make_sample_context_items


class TestPromptingModule(unittest.TestCase):
    """Kiểm tra việc xây dựng prompt tất định và cách ly metadata nội bộ."""

    def setUp(self) -> None:
        self.builder = PromptBuilder()
        self.context_items = make_sample_context_items()
        self.context = AssembledContext(
            query="câu hỏi kiểm thử",
            items=self.context_items,
            total_tokens=51,
            evidence_registry={
                it.context_item_id: it for it in self.context_items
            },
        )

    def test_deterministic_prompt_construction(self) -> None:
        """Kiểm tra việc tạo prompt là hoàn toàn tất định qua 2 lần gọi giống hệt nhau."""
        p1 = self.builder.build("CSGT dừng xe thì sao?", self.context)
        p2 = self.builder.build("CSGT dừng xe thì sao?", self.context)

        self.assertEqual(p1.user_prompt, p2.user_prompt)
        self.assertEqual(p1.system_prompt, p2.system_prompt)
        self.assertEqual(p1.prompt_version, "prompt_v1")

    def test_llm_visible_metadata_filtering(self) -> None:
        """Kiểm tra các trường nội bộ (vector score, database id) KHÔNG được xuất hiện trong prompt (Mục 7.4)."""
        prompt_req = self.builder.build("CSGT dừng xe", self.context)
        user_prompt = prompt_req.user_prompt

        # Kiểm tra nội dung công khai phải có
        self.assertIn("100/2015/NĐ-CP", user_prompt)
        self.assertIn("Điều 5", user_prompt)
        self.assertIn("[1]", user_prompt)

        # Kiểm tra metadata nội bộ tuyệt đối không được rò rỉ vào user prompt
        self.assertNotIn("internal_vector_score", user_prompt)
        self.assertNotIn("0.88", user_prompt)
        self.assertNotIn("internal_faiss_idx", user_prompt)

    def test_template_version_selection(self) -> None:
        """Kiểm tra chọn template version và báo lỗi khi version không tồn tại."""
        # Phiên bản hợp lệ
        req = self.builder.build(
            "câu hỏi", self.context, template_version="prompt_v1"
        )
        self.assertEqual(req.prompt_version, "prompt_v1")

        # Phiên bản không tồn tại -> Báo InvalidPromptError
        with self.assertRaises(InvalidPromptError):
            self.builder.build(
                "câu hỏi", self.context, template_version="non_existent_version"
            )

    def test_empty_query_raises_error(self) -> None:
        """Kiểm tra câu hỏi rỗng phải ném InvalidPromptError."""
        with self.assertRaises(InvalidPromptError):
            self.builder.build("   ", self.context)


if __name__ == "__main__":
    unittest.main()
