# src/rag/retrieval/interfaces/retriever.py
"""Interface trừu tượng cho các Retriever trong Phase 9 RAG.

Tuân thủ Mục 8: Retriever Interface:
    - 8.1 Interface: retrieve(query, retrieval_options) -> RetrievalResult
    - 8.2 Required retriever options
    - 8.4 Empty retrieval là một domain outcome hợp lệ
    - 8.5 Retrieval validation
"""

from abc import ABC, abstractmethod
from typing import List

from src.rag.retrieval.domain.models import RetrievalOptions, RetrievalResult


class Retriever(ABC):
    """Cổng giao tiếp trừu tượng cho tất cả các thành phần truy xuất trong Phase 9."""

    @abstractmethod
    def retrieve(self, query: str, options: RetrievalOptions) -> RetrievalResult:
        """Truy xuất danh sách các RetrievalItem chuẩn hóa cho câu hỏi truy vấn.

        Args:
            query: Chuỗi văn bản câu hỏi pháp lý.
            options: Cấu hình tham số truy xuất (retrieval_top_k, retrieval_mode, etc.).

        Returns:
            RetrievalResult chuẩn hóa chứa các RetrievalItem.
        """
        pass

    def batch_retrieve(
        self, queries: List[str], options: RetrievalOptions
    ) -> List[RetrievalResult]:
        """Truy xuất hàng loạt cho danh sách câu hỏi."""
        return [self.retrieve(q, options) for q in queries]
