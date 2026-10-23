"""Context-aware query rewriter for multi-turn legal conversations."""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple


FOLLOW_UP_PATTERNS = [
    r"\b(vậy|thế|thế thì|vậy thì)\b",
    r"\b(còn|nếu thế|nếu vậy|trong khi đó)\b",
    r"\b(thời gian đó|lúc này|trong lúc đó|lúc đó)\b",
    r"\b(trường hợp này|trường hợp đó|việc này|việc đó)\b",
    r"\b(điều này|điều đó|khoản này|khoản đó|quy định này)\b",
    r"\b(họ|nó|người này|người đó|đối tượng này)\b",
    r"\b(thì sao|thế nào|sao|được không|không|nhỉ)\??$",
    r"\b(ở đâu|khi nào|bao lâu|bao nhiêu)\??$",
]

QUESTION_SUFFIXES = [
    r"(\s+là\s+bao\s+lâu\??)$",
    r"(\s+là\s+bao\s+nhiêu(\s+ngày|\s+tháng)?\??)$",
    r"(\s+là\s+gì\??)$",
    r"(\s+như\s+thế\s+nào\??)$",
    r"(\s+thế\s+nào\??)$",
    r"(\s+bao\s+nhiêu\s+ngày\??)$",
    r"(\s+bao\s+nhiêu\s+tháng\??)$",
    r"(\s+bao\s+lâu\??)$",
    r"(\s+được\s+không\??)$",
    r"(\s+phải\s+làm\s+sao\??)$",
    r"(\s+ở\s+đâu\??)$",
    r"(\s+do\s+ai\s+quy\s+định\??)$",
    r"(\s+gồm\s+những\s+gì\??)$",
    r"(\s+ai\s+có\s+quyền\??)$",
    r"(\s+thì\s+sao\??)$",
    r"(\s+là\??)$",
    r"(\?+)$",
]

PREAMBLE_PATTERNS = [
    r"^(cho\s+tôi\s+hỏi\s*:?\s*)",
    r"^(tôi\s+muốn\s+hỏi\s*:?\s*)",
    r"^(xin\s+hỏi\s*:?\s*)",
    r"^(luật\s+sư\s+cho\s+hỏi\s*:?\s*)",
    r"^(cho\s+hỏi\s*:?\s*)",
    r"^(làm\s+ơn\s+cho\s+hỏi\s*:?\s*)",
]


class ConversationContextualizer:
    """Enhance follow-up queries with topic context from conversation history."""

    def __init__(self, short_query_word_threshold: int = 7) -> None:
        self.short_query_word_threshold = short_query_word_threshold

    def extract_core_topic(self, query: str) -> str:
        """Strip filler greetings and question suffixes to extract the core legal topic."""
        clean = query.strip()
        for p in PREAMBLE_PATTERNS:
            clean = re.sub(p, "", clean, flags=re.IGNORECASE).strip()
        for p in QUESTION_SUFFIXES:
            clean = re.sub(p, "", clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r"\s+là$", "", clean, flags=re.IGNORECASE).strip()
        return clean.strip()

    def is_follow_up_query(self, query: str) -> bool:
        """Check if query is dependent on prior conversation context."""
        clean = query.strip().lower()
        words = clean.split()
        if len(words) <= self.short_query_word_threshold:
            return True

        for pat in FOLLOW_UP_PATTERNS:
            if re.search(pat, clean):
                return True
        return False

    def contextualize(
        self,
        current_query: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Check history and contextualize current query if it is a follow-up.

        Returns:
            (contextualized_query, is_follow_up, prior_topic)
        """
        if not history or not current_query.strip():
            return current_query.strip(), False, None

        # Find the most recent user turn
        prior_user_query = ""
        for msg in reversed(history):
            if msg.get("role") == "user" and msg.get("content", "").strip():
                prior_user_query = msg.get("content", "").strip()
                break

        if not prior_user_query:
            return current_query.strip(), False, None

        # Check if current turn is a follow up
        if not self.is_follow_up_query(current_query):
            return current_query.strip(), False, None

        prior_topic = self.extract_core_topic(prior_user_query)
        if not prior_topic:
            return current_query.strip(), False, None

        # Avoid redundant duplication if prior topic is already in current query
        clean_current = current_query.strip()
        if prior_topic.lower() in clean_current.lower():
            return clean_current, True, prior_topic

        contextualized = f"{prior_topic}: {clean_current}"
        return contextualized, True, prior_topic

    def contextualize_with_llm(
        self,
        current_query: str,
        history: List[Dict[str, Any]],
        llm_fn: Callable[[str, str], str],
    ) -> str:
        """Rewrite follow-up query using LLM with deterministic fallback."""
        if not history or not self.is_follow_up_query(current_query):
            return current_query

        prior_user = ""
        for msg in reversed(history):
            if msg.get("role") == "user":
                prior_user = msg.get("content", "")
                break

        if not prior_user:
            return current_query

        system_prompt = (
            "Bạn là trợ lý xử lý câu hỏi pháp luật. "
            "Nhiệm vụ: Viết lại câu hỏi tiếp nối của người dùng thành 1 câu hỏi độc lập duy nhất, "
            "bổ sung đầy đủ chủ ngữ và đối tượng pháp lý từ câu trước. "
            "Chỉ trả về duy nhất 1 câu hỏi hoàn chỉnh, không giải thích gì thêm."
        )
        user_prompt = f"Câu trước: {prior_user}\nCâu tiếp nối: {current_query}\nCâu hoàn chỉnh độc lập:"

        try:
            rewritten = llm_fn(user_prompt, system_prompt).strip()
            if rewritten and len(rewritten) > 5:
                # Remove quotes if LLM added them
                rewritten = rewritten.strip('"\'')
                return rewritten
        except Exception:
            pass

        # Fallback to heuristic
        ctx_query, _, _ = self.contextualize(current_query, history)
        return ctx_query
