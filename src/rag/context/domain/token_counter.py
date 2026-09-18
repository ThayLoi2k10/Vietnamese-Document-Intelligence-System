# src/rag/context/domain/token_counter.py
"""Bộ đếm token (TokenCounter) phục vụ kiểm soát ngân sách ngữ cảnh (Context Budgeting).

Tuân thủ Mục 10.10: Token counting dependency (TokenCounter abstraction injected).
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class TokenCounter(ABC):
    """Giao diện trừu tượng cho việc tính toán số token của văn bản."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Đếm số lượng token trong đoạn văn bản.

        Args:
            text: Chuỗi văn bản cần đếm.

        Returns:
            Số lượng token nguyên dương (>= 0).
        """
        pass


class CharacterTokenCounter(TokenCounter):
    """Bộ đếm ước lượng dựa trên số ký tự (ước lượng 1 token ~ chars_per_token ký tự).

    Phù hợp cho kiểm thử đơn vị nhanh và môi trường offline không tải tokenizer cồng kềnh.
    """

    def __init__(self, chars_per_token: float = 4.0) -> None:
        self.chars_per_token = max(1.0, chars_per_token)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / self.chars_per_token))


class WhitespaceTokenCounter(TokenCounter):
    """Bộ đếm token dựa trên số từ phân tách bởi khoảng trắng."""

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(text.split())


class HuggingFaceTokenCounter(TokenCounter):
    """Bộ đếm token bọc HuggingFace PreTrainedTokenizer khi có sẵn đối tượng tokenizer."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if hasattr(self.tokenizer, "encode"):
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        return len(text.split())
