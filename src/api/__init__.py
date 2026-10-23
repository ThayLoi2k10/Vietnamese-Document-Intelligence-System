"""REST API Module for Vietnamese Legal RAG System."""

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

__all__ = [
    "LegalRAGService",
    "SearchRequest",
    "SearchResponse",
    "QueryRequest",
    "QueryResponse",
    "DocumentResponse",
    "HealthResponse",
    "RetrievedDocument",
]
