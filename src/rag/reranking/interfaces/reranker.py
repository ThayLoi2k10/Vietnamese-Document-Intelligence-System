# src/rag/reranking/interfaces/reranker.py
"""Interface trừu tượng cho các Reranker trong Phase 9 RAG.

Tuân thủ Mục 9: Reranker Interface:
    - 9.1 Interface: rerank(query, candidates, options) -> RankedResults
    - 9.2 Optionality: Hỗ trợ Reranker = None
    - 9.4 Reranker contract
"""

from abc import ABC, abstractmethod
from typing import List

from src.rag.reranking.domain.models import RankedResults, RerankOptions
from src.rag.retrieval.domain.models import RetrievalItem


class Reranker(ABC):
    """Cổng giao tiếp trừu tượng cho mọi thuật toán/mô hình Reranking."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RetrievalItem],
        options: RerankOptions,
    ) -> RankedResults:
        """Thực hiện xếp hạng lại danh sách các ứng viên từ retrieval.

        Args:
            query: Chuỗi truy vấn.
            candidates: Danh sách các RetrievalItem từ bước retrieval.
            options: Tham số rerank (reranker_top_k, v.v.).

        Returns:
            RankedResults chứa danh sách RankedCandidate với điểm số và thứ hạng mới,
            đồng thời giữ nguyên thông tin nguồn gốc.
        """
        pass
