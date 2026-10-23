"""Lớp cơ sở trừu tượng cho các kho lưu trữ và chỉ mục vector."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Union
import numpy as np

from src.retrieval.indexing.models import VectorPoint, VectorSearchResult


class BaseVectorStore(ABC):
    def __init__(self, dimension: int = 1024, collection_name: str = "legal_chunks") -> None:
        self.dimension = dimension
        self.collection_name = collection_name

    @abstractmethod
    def upsert(self, points: List[VectorPoint], batch_size: int = 256) -> int:
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 100,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        pass

    @abstractmethod
    def get(self, point_ids: List[str]) -> List[Optional[VectorPoint]]:
        pass

    @abstractmethod
    def delete(self, point_ids: List[str]) -> int:
        pass

    @abstractmethod
    def count(self, filter_dict: Optional[Dict[str, Any]] = None) -> int:
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> Self:
        pass
