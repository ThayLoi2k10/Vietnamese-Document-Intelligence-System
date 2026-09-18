# src/retrieval/chunking/__init__.py
"""Package quản lý và triển khai các chiến lược phân đoạn văn bản pháp luật (Phase 7)."""

from src.retrieval.chunking.base import BaseChunker
from src.retrieval.chunking.fixed_size import FixedSizeChunker
from src.retrieval.chunking.models import (
    LegalChunk,
    compute_content_hash,
    generate_deterministic_chunk_id,
)
from src.retrieval.chunking.paragraph import ParagraphChunker
from src.retrieval.chunking.recursive import RecursiveChunker
from src.retrieval.chunking.sentence import SentenceChunker
from src.retrieval.chunking.structure_aware import StructureAwareChunker

__all__ = [
    "BaseChunker",
    "LegalChunk",
    "compute_content_hash",
    "generate_deterministic_chunk_id",
    "FixedSizeChunker",
    "RecursiveChunker",
    "SentenceChunker",
    "ParagraphChunker",
    "StructureAwareChunker",
]
