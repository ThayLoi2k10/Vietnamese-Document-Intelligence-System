# src/retrieval/dense/__init__.py
"""Dense Retrieval module for Vietnamese Legal Document Intelligence (Phase 6).

Cung cấp:
    - DenseEncoder: Mã hóa câu truy vấn và văn bản điều luật bằng BAAI/bge-m3.
    - DenseIndex: Quản lý chỉ mục vector chính xác FAISS IndexFlatIP.
    - DenseRetriever: Lớp truy xuất kế thừa BaseRetriever chuẩn của dự án.
    - DenseArtifactManager: Quản lý lưu trữ, fingerprint và tính toàn vẹn artifact.
"""

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
