# src/retrieval/indexing/base.py
"""Lớp trừu tượng BaseVectorStore cho kiến trúc chỉ mục vector trong Phase 8.

Tuân thủ nghiêm ngặt:
    - Mục 5: Functional Requirements (FR-1 đến FR-12)
    - Mục 24: Query Interface Contract
    - Mục 25: Filtering Contract (pre-filter)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Union
import numpy as np

from src.retrieval.indexing.models import VectorPoint, VectorSearchResult


class BaseVectorStore(ABC):
    """Giao diện trừu tượng cơ sở cho các hệ quản trị lưu trữ vector."""

    def __init__(self, dimension: int = 1024, collection_name: str = "legal_chunks") -> None:
        self.dimension = dimension
        self.collection_name = collection_name

    @abstractmethod
    def upsert(self, points: List[VectorPoint], batch_size: int = 256) -> int:
        """Thêm mới hoặc cập nhật hàng loạt điểm vector một cách tất định (idempotent upsert)."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 100,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Tìm kiếm Top-K chunk tương đồng nhất kèm lọc metadata (pre-filtering)."""
        pass

    @abstractmethod
    def get(self, point_ids: List[str]) -> List[Optional[VectorPoint]]:
        """Lấy thông tin điểm vector theo danh sách point_id."""
        pass

    @abstractmethod
    def delete(self, point_ids: List[str]) -> int:
        """Xóa hoặc vô hiệu hóa các điểm vector theo danh sách point_id."""
        pass

    @abstractmethod
    def count(self, filter_dict: Optional[Dict[str, Any]] = None) -> int:
        """Đếm tổng số điểm vector hiện có (kèm điều kiện lọc nếu có)."""
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Lưu trữ chỉ mục / snapshot xuống đĩa."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> Self:
        """Nạp chỉ mục / kết nối lại kho lưu trữ từ đĩa."""
        pass
