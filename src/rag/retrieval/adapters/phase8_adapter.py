"""Adapter transforming Phase 8 retrieval outputs into Phase 9 domain models."""

from typing import Any, Callable, Dict, List, Optional

from src.rag.errors.exceptions import (
    InvalidRetrievalResultError,
    MissingRequiredMetadataError,
)
from src.rag.retrieval.domain.models import (
    ArticleResult,
    RetrievalItem,
    RetrievalMetadata,
    RetrievalMode,
    RetrievalOptions,
    RetrievalResult,
)
from src.rag.retrieval.interfaces.retriever import Retriever


class Phase8RetrievalAdapter(Retriever):
    """Wraps Phase 8 search backends and normalizes data to Phase 9 contracts."""

    def __init__(
        self,
        search_fn: Optional[Callable[[str, int], List[Any]]] = None,
        retriever_name: str = "phase8_vector_store",
        component_version: str = "v1.0.0",
        vector_store: Optional[Any] = None,
        encoder: Optional[Any] = None,
    ) -> None:
        self.search_fn = search_fn
        self.retriever_name = retriever_name
        self.component_version = component_version
        self.vector_store = vector_store
        self.encoder = encoder

    def normalize_phase8_record(
        self, raw_item: Any, expected_rank: int
    ) -> RetrievalItem:
        if isinstance(raw_item, dict):
            payload = raw_item.get("payload", {})
            chunk_id = raw_item.get("chunk_id") or payload.get("chunk_id")
            global_article_id = raw_item.get("global_article_id") or payload.get("global_article_id")
            score = raw_item.get("score")
            rank = raw_item.get("rank", expected_rank)
            law_id = payload.get("law_id") or raw_item.get("law_id")
            article_identifier = payload.get("article_identifier") or raw_item.get("article_identifier")
            chunk_text = (
                payload.get("chunk_text")
                or raw_item.get("chunk_text")
                or raw_item.get("text")
            )
            chunk_order = payload.get("chunk_order") or raw_item.get("chunk_order")
            total_chunks = payload.get("total_chunks") or raw_item.get("total_chunks")
            token_count = payload.get("token_count") or raw_item.get("token_count")
            extra_metadata = {
                k: v
                for k, v in payload.items()
                if k not in {
                    "chunk_id", "global_article_id", "law_id", "article_identifier",
                    "chunk_text", "chunk_order", "total_chunks", "token_count",
                }
            }
        else:
            chunk_id = getattr(raw_item, "chunk_id", None)
            global_article_id = getattr(raw_item, "global_article_id", None)
            score = getattr(raw_item, "score", None)
            rank = getattr(raw_item, "rank", expected_rank)
            payload = getattr(raw_item, "payload", {}) or {}

            law_id = payload.get("law_id") or getattr(raw_item, "law_id", None)
            article_identifier = payload.get("article_identifier") or getattr(raw_item, "article_identifier", None)
            chunk_text = (
                payload.get("chunk_text")
                or getattr(raw_item, "chunk_text", None)
                or getattr(raw_item, "text", None)
            )
            chunk_order = payload.get("chunk_order") or getattr(raw_item, "chunk_order", None)
            total_chunks = payload.get("total_chunks") or getattr(raw_item, "total_chunks", None)
            token_count = payload.get("token_count") or getattr(raw_item, "token_count", None)
            extra_metadata = {
                k: v
                for k, v in payload.items()
                if k not in {
                    "chunk_id", "global_article_id", "law_id", "article_identifier",
                    "chunk_text", "chunk_order", "total_chunks", "token_count",
                }
            }

        if not law_id or not str(law_id).strip():
            raise MissingRequiredMetadataError(
                "Phase 8 record missing required 'law_id'.",
                details={"raw_item": str(raw_item)},
            )

        if chunk_text is None or not str(chunk_text).strip():
            raise InvalidRetrievalResultError(
                "Phase 8 record missing valid 'chunk_text'.",
                details={"raw_item": str(raw_item)},
            )

        if rank < 1:
            raise InvalidRetrievalResultError(
                f"Phase 8 record has invalid rank: {rank}",
                details={"raw_item": str(raw_item)},
            )

        return RetrievalItem(
            chunk_id=str(chunk_id) if chunk_id is not None else None,
            global_article_id=(
                int(global_article_id) if global_article_id is not None else None
            ),
            law_id=str(law_id),
            article_identifier=(
                str(article_identifier) if article_identifier is not None else None
            ),
            text=str(chunk_text),
            chunk_order=int(chunk_order) if chunk_order is not None else None,
            total_chunks=int(total_chunks) if total_chunks is not None else None,
            score=float(score) if score is not None else None,
            rank=int(rank),
            token_count=int(token_count) if token_count is not None else None,
            source_metadata=extra_metadata,
        )

    def retrieve(self, query: str, options: RetrievalOptions) -> RetrievalResult:
        top_k = options.retrieval_top_k
        if self.search_fn is not None:
            raw_items = self.search_fn(query, top_k)
        elif self.vector_store is not None:
            if self.encoder is None:
                from src.retrieval.dense.encoder import DenseEncoder
                self.encoder = DenseEncoder()
            q_vec = self.encoder.encode_queries([query], batch_size=1)
            raw_items = self.vector_store.search(q_vec, limit=top_k)
        else:
            raise InvalidRetrievalResultError(
                "Phase8RetrievalAdapter requires search_fn or vector_store."
            )

        items: List[RetrievalItem] = []
        for idx, raw_item in enumerate(raw_items, start=1):
            item = self.normalize_phase8_record(raw_item, expected_rank=idx)
            item.validate(RetrievalMode.CHUNK)
            items.append(item)

        aggregated: Dict[str, ArticleResult] = {}
        for it in items:
            key = (
                str(it.global_article_id)
                if it.global_article_id is not None
                else f"{it.law_id}_{it.article_identifier}"
            )
            if key not in aggregated:
                aggregated[key] = ArticleResult(
                    law_id=it.law_id,
                    article_rank=len(aggregated) + 1,
                    global_article_id=it.global_article_id,
                    article_identifier=it.article_identifier,
                    best_chunk_score=it.score,
                    contributing_chunks=[it.chunk_id] if it.chunk_id else [],
                )
            else:
                art = aggregated[key]
                if it.chunk_id:
                    art.contributing_chunks.append(it.chunk_id)
                if (
                    it.score is not None
                    and art.best_chunk_score is not None
                    and it.score > art.best_chunk_score
                ):
                    art.best_chunk_score = it.score

        metadata = RetrievalMetadata(
            retriever_name=self.retriever_name,
            retrieval_mode=RetrievalMode.CHUNK,
            top_k=top_k,
            component_version=self.component_version,
            parameters=options.parameters,
        )

        res = RetrievalResult(
            query=query,
            retrieval_mode=RetrievalMode.CHUNK,
            items=items,
            aggregated_articles=list(aggregated.values()),
            retriever_metadata=metadata,
        )
        res.validate()
        return res
