"""Package cung cấp các mô hình truy xuất và xếp hạng văn bản pháp luật."""

from src.retrieval.base import BaseRetriever, RetrievalResult, extract_corpus_columns
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.tfidf_retriever import TfidfRetriever

__all__ = [
    "BaseRetriever",
    "RetrievalResult",
    "extract_corpus_columns",
    "TfidfRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "ChunkedVectorRetriever",
    "ArticleReranker",
]


def __getattr__(name: str):
    if name == "HybridRetriever":
        from src.retrieval.hybrid.retriever import HybridRetriever
        return HybridRetriever
    if name == "ChunkedVectorRetriever":
        from src.retrieval.chunked_retriever import ChunkedVectorRetriever
        return ChunkedVectorRetriever
    if name == "ArticleReranker":
        from src.retrieval.reranking.cross_encoder import ArticleReranker
        return ArticleReranker
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")