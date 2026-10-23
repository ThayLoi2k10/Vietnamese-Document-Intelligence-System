"""Unit tests for multi-turn conversation contextualizer."""

import unittest
from src.rag.query.contextualizer import ConversationContextualizer


class TestConversationContextualizer(unittest.TestCase):
    def setUp(self):
        self.contextualizer = ConversationContextualizer()

    def test_extract_core_topic(self):
        q1 = "Cho tôi hỏi thời gian thử việc tối đa của người lao động là bao lâu?"
        topic1 = self.contextualizer.extract_core_topic(q1)
        self.assertEqual(topic1, "thời gian thử việc tối đa của người lao động")

        q2 = "Luật sư cho hỏi thủ tục sang tên sổ đỏ đất nông nghiệp như thế nào?"
        topic2 = self.contextualizer.extract_core_topic(q2)
        self.assertEqual(topic2, "thủ tục sang tên sổ đỏ đất nông nghiệp")

    def test_is_follow_up_query(self):
        # Short query
        self.assertTrue(self.contextualizer.is_follow_up_query("Lệ phí bao nhiêu?"))
        # Contains pronoun / reference
        self.assertTrue(self.contextualizer.is_follow_up_query("Trong thời gian đó có được trả lương không?"))
        self.assertTrue(self.contextualizer.is_follow_up_query("Vậy nếu công ty vi phạm thì sao?"))
        self.assertTrue(self.contextualizer.is_follow_up_query("Còn đối với người quản lý doanh nghiệp thì sao?"))

        # Standalone full query should NOT be considered a follow up
        long_standalone = "Thủ tục đăng ký thành lập doanh nghiệp tư nhân theo quy định của Luật Doanh nghiệp"
        self.assertFalse(self.contextualizer.is_follow_up_query(long_standalone))

    def test_first_turn_unchanged(self):
        q = "Thời gian thử việc tối đa là bao lâu?"
        res, is_follow_up, prior = self.contextualizer.contextualize(q, history=[])
        self.assertEqual(res, q)
        self.assertFalse(is_follow_up)
        self.assertIsNone(prior)

    def test_contextualize_follow_up(self):
        history = [
            {"role": "user", "content": "Thời gian thử việc tối đa của người lao động là bao lâu?"},
            {"role": "assistant", "content": "Theo Điều 25 Bộ luật Lao động 2019, thời gian thử việc tối đa là 60 ngày..."},
        ]
        q_followup = "Vậy trong thời gian đó có được trả lương không?"
        res, is_follow_up, prior = self.contextualizer.contextualize(q_followup, history=history)

        self.assertTrue(is_follow_up)
        self.assertEqual(prior.lower(), "thời gian thử việc tối đa của người lao động")
        self.assertIn("thời gian thử việc tối đa của người lao động", res.lower())
        self.assertIn("Vậy trong thời gian đó có được trả lương không?", res)

    def test_independent_query_in_history_not_overwritten(self):
        history = [
            {"role": "user", "content": "Thời gian thử việc tối đa là bao lâu?"},
            {"role": "assistant", "content": "Tối đa 60 ngày theo Điều 25."},
        ]
        q_unrelated = "Thủ tục sang tên sổ đỏ tại văn phòng đăng ký đất đai gồm những giấy tờ gì"
        res, is_follow_up, prior = self.contextualizer.contextualize(q_unrelated, history=history)

        self.assertFalse(is_follow_up)
        self.assertEqual(res, q_unrelated)

    def test_llm_fallback(self):
        def failing_llm(u, s):
            raise ConnectionError("LLM offline")

        history = [
            {"role": "user", "content": "Thời gian thử việc của người lao động là bao lâu?"},
            {"role": "assistant", "content": "Tối đa 60 ngày..."},
        ]
        q = "Có được hưởng lương không?"
        res = self.contextualizer.contextualize_with_llm(q, history, failing_llm)
        self.assertIn("thời gian thử việc của người lao động", res.lower())


if __name__ == "__main__":
    unittest.main()
