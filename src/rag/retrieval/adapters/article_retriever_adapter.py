# src/rag/retrieval/adapters/article_retriever_adapter.py
"""Adapter cho các Retriever cấp độ Article (Phase 5/6 BaseRetriever) sang domain contract Phase 9.

Tuân thủ Mục 2.3:
    - Phase 9 giao tiếp qua abstraction có thể tiêu thụ chunk-level hoặc article-level retrieval.
    - Khi article-level retrieval được sử dụng, Phase 9 không ép buộc chunk architecture.
"""

from typing import Any, Callable, Dict, List, Optional

from src.rag.errors.exceptions import (
    InvalidRetrievalResultError,
    MissingRequiredMetadataError,
)
from src.rag.retrieval.domain.models import (
    RetrievalItem,
    RetrievalMetadata,
    RetrievalMode,
    RetrievalOptions,
    RetrievalResult,
)
from src.rag.retrieval.interfaces.retriever import Retriever


class ArticleRetrieverAdapter(Retriever):
    """Adapter tích hợp các retriever cấp độ Article (Phase 5/6) vào Phase 9."""

    def __init__(
        self,
        search_fn: Optional[Callable[[str, int], List[Any]]] = None,
        article_lookup: Optional[Dict[int, Dict[str, Any]]] = None,
        retriever_name: str = "article_retriever",
        component_version: str = "v1.0.0",
    ) -> None:
        """Khởi tạo adapter.

        Args:
            search_fn: Hàm nhận (query, top_k) và trả về danh sách RetrievalResult (từ src/retrieval/base.py)
                       chứa article_id, score, rank.
            article_lookup: Bảng tra cứu thông tin chi tiết của article (law_id, text, article_identifier, v.v.).
            retriever_name: Tên retriever.
            component_version: Phiên bản thành phần.
        """
        self.search_fn = search_fn
        self.article_lookup = article_lookup or {}
        self.retriever_name = retriever_name
        self.component_version = component_version

    def retrieve(self, query: str, options: RetrievalOptions) -> RetrievalResult:
        if self.search_fn is None:
            raise InvalidRetrievalResultError(
                "ArticleRetrieverAdapter chưa được cấu hình search_fn."
            )

        top_k = options.retrieval_top_k
        raw_results = self.search_fn(query, top_k)

        items: List[RetrievalItem] = []
        for idx, r in enumerate(raw_results, start=1):
            if isinstance(r, dict):
                art_id = r.get("article_id") or r.get("global_article_id")
                score = r.get("score")
                rank = r.get("rank", idx)
            else:
                art_id = getattr(r, "article_id", None)
                if art_id is None:
                    art_id = getattr(r, "global_article_id", None)
                score = getattr(r, "score", None)
                rank = getattr(r, "rank", idx)

            if art_id is None:
                raise MissingRequiredMetadataError(
                    "Kết quả truy xuất thiếu article_id.",
                    details={"raw_item": str(r)},
                )

            art_info = self.article_lookup.get(int(art_id), {})
            law_id = art_info.get("law_id") or f"law_doc_{art_id}"
            text = art_info.get("text") or art_info.get("article_content") or ""
            article_identifier = art_info.get("article_identifier") or f"Điều {art_id}"

            item = RetrievalItem(
                global_article_id=int(art_id),
                law_id=str(law_id),
                article_identifier=str(article_identifier),
                text=str(text),
                score=float(score) if score is not None else None,
                rank=int(rank),
                source_metadata={
                    k: v
                    for k, v in art_info.items()
                    if k not in {"law_id", "text", "article_identifier"}
                },
            )
            item.validate(RetrievalMode.ARTICLE)
            items.append(item)

        metadata = RetrievalMetadata(
            retriever_name=self.retriever_name,
            retrieval_mode=RetrievalMode.ARTICLE,
            top_k=top_k,
            component_version=self.component_version,
            parameters=options.parameters,
        )

        res = RetrievalResult(
            query=query,
            retrieval_mode=RetrievalMode.ARTICLE,
            items=items,
            retriever_metadata=metadata,
        )
        res.validate()
        return res
