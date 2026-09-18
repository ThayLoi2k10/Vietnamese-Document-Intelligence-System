# src/retrieval/indexing/__init__.py
"""Package quản lý và lưu trữ chỉ mục vector bền vững (Phase 8)."""

from src.retrieval.indexing.base import BaseVectorStore
from src.retrieval.indexing.faiss_vector_store import FaissVectorStore
from src.retrieval.indexing.lifecycle import IncrementalIngestionManager, IngestionRecord, IngestionState
from src.retrieval.indexing.models import (
    VectorPoint,
    VectorSearchResult,
    generate_deterministic_point_id,
)
from src.retrieval.indexing.qdrant_store import QdrantVectorStore
from src.retrieval.indexing.rebuild import rebuild_vector_index
from src.retrieval.indexing.reconciliation import ReconciliationReport, run_reconciliation

__all__ = [
    "BaseVectorStore",
    "FaissVectorStore",
    "QdrantVectorStore",
    "VectorPoint",
    "VectorSearchResult",
    "generate_deterministic_point_id",
    "IncrementalIngestionManager",
    "IngestionRecord",
    "IngestionState",
    "ReconciliationReport",
    "run_reconciliation",
    "rebuild_vector_index",
]
