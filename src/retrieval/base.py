"""Lớp cơ sở và cấu trúc dữ liệu chuẩn cho các mô hình truy xuất văn bản pháp luật."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Self, Union
import pandas as pd


@dataclass(frozen=True)
class RetrievalResult:
    article_id: int
    score: float
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_corpus_columns(
    corpus: Union[pd.DataFrame, List[Dict[str, Any]]],
) -> tuple[list[int], list[str]]:
    df = pd.DataFrame(corpus) if isinstance(corpus, list) else corpus

    if "global_article_id" in df.columns:
        id_col = "global_article_id"
    elif "article_id" in df.columns:
        id_col = "article_id"
    else:
        raise KeyError("Không tìm thấy cột định danh 'global_article_id' hoặc 'article_id' trong corpus.")

    for candidate in ("text", "article_content", "content"):
        if candidate in df.columns:
            text_col = candidate
            break
    else:
        raise KeyError("Không tìm thấy cột văn bản ('text', 'article_content', 'content') trong corpus.")

    article_ids = df[id_col].astype(int).tolist()
    texts = df[text_col].fillna("").astype(str).tolist()
    return article_ids, texts


class BaseRetriever(ABC):
    def __init__(self, **kwargs: Any) -> None:
        self.is_fitted: bool = False

    def _check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} chưa được fit hoặc load chỉ mục. "
                "Hãy gọi method fit() hoặc load() trước khi thực hiện tìm kiếm."
            )

    @abstractmethod
    def fit(self, corpus: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Self:
        pass

    @abstractmethod
    def search(self, query_text: str, top_k: int = 10) -> List[RetrievalResult]:
        pass

    def batch_search(
        self, queries: List[str], top_k: int = 10
    ) -> List[List[RetrievalResult]]:
        return [self.search(q, top_k=top_k) for q in queries]

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> Self:
        pass