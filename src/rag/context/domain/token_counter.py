"""Token counter implementations for context budgeting."""

from abc import ABC, abstractmethod
from typing import Any


class TokenCounter(ABC):
    """Abstract interface for token count estimation."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass


class CharacterTokenCounter(TokenCounter):
    """Character-ratio token counter for fast offline estimation."""

    def __init__(self, chars_per_token: float = 4.0) -> None:
        self.chars_per_token = max(1.0, chars_per_token)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / self.chars_per_token))


class WhitespaceTokenCounter(TokenCounter):
    """Whitespace-delimited word token counter."""

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(text.split())


class HuggingFaceTokenCounter(TokenCounter):
    """Token counter wrapping a HuggingFace tokenizer instance."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if hasattr(self.tokenizer, "encode"):
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(text.split())
