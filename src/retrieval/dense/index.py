"""Quản lý chỉ mục tìm kiếm vector chính xác FAISS IndexFlatIP cho văn bản pháp luật."""

import json
from pathlib import Path
from typing import List, Tuple, Union
import faiss
import numpy as np

EMBEDDING_DIMENSION = 1024


class DenseIndex:
    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        if dimension <= 0:
            raise ValueError(f"Kích thước vector phải dương, nhận được: {dimension}")
        self.dimension = dimension
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.corpus_ids: List[int] = []

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def add(self, embeddings: np.ndarray, corpus_ids: List[int]) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings phải là mảng 2 chiều, nhận được: {embeddings.shape}")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Kích thước vector ({embeddings.shape[1]}) không khớp với index ({self.dimension})."
            )
        if embeddings.shape[0] != len(corpus_ids):
            raise ValueError(
                f"Số lượng vector ({embeddings.shape[0]}) không khớp số lượng ID ({len(corpus_ids)})."
            )

        if not embeddings.flags["C_CONTIGUOUS"]:
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        elif embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        self.index.add(embeddings)
        self.corpus_ids.extend([int(cid) for cid in corpus_ids])

        if self.ntotal != len(self.corpus_ids):
            raise RuntimeError(
                f"Lỗi đồng bộ: FAISS ntotal={self.ntotal} khác với len(corpus_ids)={len(self.corpus_ids)}."
            )

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 100,
    ) -> Tuple[np.ndarray, List[List[int]]]:
        if top_k <= 0:
            raise ValueError(f"top_k phải > 0, nhận được: {top_k}")
        if self.ntotal == 0:
            raise RuntimeError("Chỉ mục đang rỗng, không thể tìm kiếm.")

        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        if query_embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Kích thước query ({query_embeddings.shape[1]}) không khớp với index ({self.dimension})."
            )

        if not query_embeddings.flags["C_CONTIGUOUS"]:
            query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        elif query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)

        effective_k = min(top_k, self.ntotal)
        scores, row_indices = self.index.search(query_embeddings, effective_k)

        mapped_ids: List[List[int]] = []
        for row in row_indices:
            q_ids: List[int] = []
            seen = set()
            for idx in row:
                if 0 <= idx < len(self.corpus_ids):
                    article_id = self.corpus_ids[idx]
                    if article_id not in seen:
                        q_ids.append(article_id)
                        seen.add(article_id)
            mapped_ids.append(q_ids)

        return scores, mapped_ids

    def save(self, index_file: Union[str, Path], ids_file: Union[str, Path]) -> None:
        index_path = Path(index_file)
        ids_path = Path(ids_file)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        ids_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(self.corpus_ids, f)

    @classmethod
    def load(cls, index_file: Union[str, Path], ids_file: Union[str, Path]) -> "DenseIndex":
        index_path = Path(index_file)
        ids_path = Path(ids_file)

        if not index_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file chỉ mục FAISS tại: {index_path}")
        if not ids_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file danh sách ID tại: {ids_path}")

        faiss_index = faiss.read_index(str(index_path))
        with open(ids_path, "r", encoding="utf-8") as f:
            corpus_ids = json.load(f)

        instance = cls(dimension=faiss_index.d)
        instance.index = faiss_index
        instance.corpus_ids = [int(cid) for cid in corpus_ids]

        if instance.ntotal != len(instance.corpus_ids):
            raise ValueError(
                f"Chỉ mục không hợp lệ: FAISS ntotal ({instance.ntotal}) "
                f"không bằng len(corpus_ids) ({len(instance.corpus_ids)})."
            )

        return instance
