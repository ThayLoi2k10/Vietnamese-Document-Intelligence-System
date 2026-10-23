"""FastAPI route definitions for Vietnamese Legal RAG system."""

import time
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.schemas import (
    DocumentResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrievedDocument,
    SearchRequest,
    SearchResponse,
)
from src.api.service import LegalRAGService

router = APIRouter(prefix="/api/v1", tags=["Legal RAG API"])

# Global singleton or dependency container
_rag_service_instance: LegalRAGService | None = None


def get_rag_service() -> LegalRAGService:
    """FastAPI dependency to retrieve or initialize LegalRAGService."""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = LegalRAGService(lazy_load=True)
    return _rag_service_instance


def set_rag_service(service: LegalRAGService) -> None:
    """Override service instance for testing or custom configuration."""
    global _rag_service_instance
    _rag_service_instance = service


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health & Status Check",
    description="Returns indexing status, document count, and uptime.",
)
def health_check(
    service: Annotated[LegalRAGService, Depends(get_rag_service)],
) -> HealthResponse:
    uptime = time.time() - service.start_time
    indices_status = {
        "bm25": service.bm25_retriever is not None,
        "dense": service.dense_retriever is not None,
        "reranker": service.reranker is not None,
    }
    is_loaded = getattr(service, "_is_loaded", True)
    status_str = "healthy" if (is_loaded or indices_status["bm25"]) else "loading"
    return HealthResponse(
        status=status_str,
        version="1.0.0",
        total_articles=len(service.corpus),
        indices_loaded=indices_status,
        default_backend="auto",
        uptime_seconds=round(uptime, 2),
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Retrieve Top-K Legal Documents",
    description="Executes lexical, dense, hybrid, or reranked search across Vietnamese legal articles.",
)
def search_documents(
    payload: SearchRequest,
    service: Annotated[LegalRAGService, Depends(get_rag_service)],
) -> SearchResponse:
    raw_docs, norm_query, latency_ms = service.search(
        query=payload.query,
        retriever_type=payload.retriever_type,
        top_k=payload.top_k,
    )
    docs = [RetrievedDocument(**d) for d in raw_docs]
    return SearchResponse(
        query=payload.query,
        normalized_query=norm_query,
        total_results=len(docs),
        retriever_type=payload.retriever_type,
        documents=docs,
        latency_ms=round(latency_ms, 2),
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="End-to-End Legal Question Answering",
    description="Contextualizes question, retrieves relevant articles, synthesizes answer via LLM, and verifies grounding.",
)
def answer_legal_query(
    payload: QueryRequest,
    service: Annotated[LegalRAGService, Depends(get_rag_service)],
) -> QueryResponse:
    try:
        res = service.answer_query(
            query=payload.query,
            conversation_history=payload.conversation_history,
            backend=payload.backend,
            colab_url=payload.colab_url,
            api_key=payload.api_key,
            model=payload.model,
            retriever_type=payload.retriever_type,
            top_k=payload.top_k,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
        )
        return QueryResponse(**res)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi xử lý câu hỏi: {str(exc)}",
        ) from exc


@router.get(
    "/documents/{article_id}",
    response_model=DocumentResponse,
    summary="Get Legal Article by ID",
    description="Fetches full text and metadata for a specific global article ID.",
)
def get_document_by_id(
    article_id: int,
    service: Annotated[LegalRAGService, Depends(get_rag_service)],
) -> DocumentResponse:
    doc = service.get_document(article_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy điều luật với ID: {article_id}",
        )
    return DocumentResponse(**doc)
