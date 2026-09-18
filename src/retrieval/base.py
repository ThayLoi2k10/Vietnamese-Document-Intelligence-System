from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Self, Union
import pandas as pd


@dataclass(frozen=True)
class RetrievalResult:
    """Đặc tả kết quả truy xuất cho một tài liệu đơn lẻ.

    Attributes:
        article_id: ID định danh duy nhất của đơn vị văn bản trong corpus.
        score: Điểm số tương đồng (BM25 score, Cosine similarity, hoặc Dense logit).
        rank: Thứ hạng của kết quả (1-indexed, 1 là độ liên quan cao nhất).
    """

    article_id: int
    score: float
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseRetriever(ABC):
    """Abstract Base Class định nghĩa giao diện chung cho mọi Retriever."""

    def __init__(self, **kwargs: Any) -> None:
        self.is_fitted: bool = False

    def _check_is_fitted(self) -> None:
        """Kiểm tra chỉ mục đã được huấn luyện hoặc tải lên chưa trước khi thao tác."""
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} chưa được fit hoặc load chỉ mục. "
                "Hãy gọi method fit() hoặc load() trước khi thực hiện tìm kiếm."
            )

    @abstractmethod
    def fit(self, corpus: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Self:
        """Nạp corpus, xây dựng chỉ mục (Inverted Index, Matrix, hoặc Vector Index).

        Args:
            corpus: DataFrame hoặc danh sách bản ghi chứa các trường:
                    - 'article_id' hoặc 'global_article_id' (int)
                    - 'text' hoặc 'article_content' (str)

        Returns:
            self: Trả về chính đối tượng retriever sau khi fit.
        """
        pass

    @abstractmethod
    def search(self, query_text: str, top_k: int = 10) -> List[RetrievalResult]:
        """Truy vấn tìm kiếm Top-k tài liệu liên quan nhất cho một câu hỏi.

        Args:
            query_text: Chuỗi văn bản truy vấn.
            top_k: Số lượng tài liệu tối đa cần trả về (top_k > 0).

        Returns:
            Danh sách các đối tượng RetrievalResult được sắp xếp giảm dần theo score.
        """
        pass

    def batch_search(
        self, queries: List[str], top_k: int = 10
    ) -> List[List[RetrievalResult]]:
        """Truy vấn hàng loạt (Batch Search) để tối ưu hóa thời gian chạy evaluation.

        Args:
            queries: Danh sách các chuỗi truy vấn.
            top_k: Số lượng tài liệu tối đa cho mỗi query.

        Returns:
            Danh sách danh sách RetrievalResult tương ứng với thứ tự đầu vào của queries.
        """
        return [self.search(q, top_k=top_k) for q in queries]

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Lưu trữ chỉ mục (index) và cấu hình mô hình xuống ổ đĩa."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> Self:
        """Nạp chỉ mục và khôi phục retriever từ ổ đĩa."""
        pass