"""Package truy xuất văn bản ngữ nghĩa sử dụng dense bi-encoder và FAISS flat index."""

from src.retrieval.dense.artifacts import DenseArtifactManager
from src.retrieval.dense.encoder import DenseEncoder
from src.retrieval.dense.index import DenseIndex
from src.retrieval.dense.retriever import DenseRetriever

__all__ = [
    "DenseEncoder",
    "DenseIndex",
    "DenseRetriever",
    "DenseArtifactManager",
]
