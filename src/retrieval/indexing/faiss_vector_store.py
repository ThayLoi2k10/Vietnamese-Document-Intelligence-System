# src/retrieval/indexing/faiss_vector_store.py
"""Chỉ mục tham chiếu chính xác FAISS Flat IP kèm quản lý payload cho Phase 8.

Tuân thủ Mục 8.1 và Mục 10.1 của Đặc tả Phase 8:
    - Sử dụng faiss.IndexFlatIP làm 'correctness oracle' (chuẩn đối sánh độ chính xác tuyệt đối).
    - Lưu trữ payload metadata và hỗ trợ lọc metadata chính xác (pre-filtering).
    - Hỗ trợ lưu trữ persistent và phục hồi hoàn toàn trạng thái.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Union
import faiss
import numpy as np

from src.retrieval.indexing.base import BaseVectorStore
from src.retrieval.indexing.models import VectorPoint, VectorSearchResult


class FaissVectorStore(BaseVectorStore):
    """Kho lưu trữ vector FAISS Flat IP chính xác kèm quản lý payload trong bộ nhớ."""

    def __init__(self, dimension: int = 1024, collection_name: str = "faiss_exact_reference") -> None:
        super().__init__(dimension=dimension, collection_name=collection_name)
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.point_ids: List[str] = []
        self.payloads: Dict[str, Dict[str, Any]] = {}
        self.vectors: Dict[str, np.ndarray] = {}

    def _matches_filter(self, payload: Dict[str, Any], filter_dict: Optional[Dict[str, Any]]) -> bool:
        """Kiểm tra điều kiện pre-filter trên payload metadata."""
        if not filter_dict:
            return True
        for k, v in filter_dict.items():
            if payload.get(k) != v:
                return False
        return True

    def upsert(self, points: List[VectorPoint], batch_size: int = 256) -> int:
        """Thêm mới hoặc cập nhật các điểm vector vào kho."""
        if not points:
            return 0

        # Cập nhật thông tin điểm trong từ điển
        for p in points:
            pid = p.point_id
            self.payloads[pid] = p.to_payload()
            vec = np.array(p.vector, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            self.vectors[pid] = vec

        # Tái tạo lại FAISS index để đảm bảo tính đồng bộ hoàn hảo
        self.point_ids = list(self.vectors.keys())
        if self.point_ids:
            all_vecs = np.stack([self.vectors[pid] for pid in self.point_ids], axis=0)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(np.ascontiguousarray(all_vecs, dtype=np.float32))

        return len(points)

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 100,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Tìm kiếm tương đồng Inner Product chính xác, có hỗ trợ pre-filtering."""
        if self.index.ntotal == 0:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec /= norm

        # Nếu có điều kiện lọc metadata: quét chính xác các ứng viên thỏa mãn filter (pre-filter)
        if filter_dict:
            matched_indices = []
            matched_pids = []
            for idx, pid in enumerate(self.point_ids):
                payload = self.payloads.get(pid, {})
                if self._matches_filter(payload, filter_dict):
                    matched_indices.append(idx)
                    matched_pids.append(pid)

            if not matched_indices:
                return []

            sub_vecs = np.stack([self.vectors[pid] for pid in matched_pids], axis=0)
            sub_scores = np.dot(sub_vecs, q_vec.T).flatten()

            top_indices = np.argsort(-sub_scores)[:limit]
            results: List[VectorSearchResult] = []
            for rank_i, sub_idx in enumerate(top_indices, start=1):
                pid = matched_pids[sub_idx]
                score = float(sub_scores[sub_idx])
                payload = self.payloads.get(pid, {})
                results.append(
                    VectorSearchResult(
                        point_id=pid,
                        chunk_id=payload.get("chunk_id", pid),
                        global_article_id=int(payload.get("global_article_id", -1)),
                        score=score,
                        rank=rank_i,
                        payload=payload,
                    )
                )
            return results

        # Tìm kiếm trực tiếp qua FAISS index
        eff_k = min(limit, self.index.ntotal)
        scores, indices = self.index.search(q_vec, eff_k)

        results = []
        for rank_i, (idx, sc) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            pid = self.point_ids[idx]
            payload = self.payloads.get(pid, {})
            results.append(
                VectorSearchResult(
                    point_id=pid,
                    chunk_id=payload.get("chunk_id", pid),
                    global_article_id=int(payload.get("global_article_id", -1)),
                    score=float(sc),
                    rank=rank_i,
                    payload=payload,
                )
            )

        return results

    def get(self, point_ids: List[str]) -> List[Optional[VectorPoint]]:
        """Truy vấn các điểm vector theo danh sách point_id."""
        ret = []
        for pid in point_ids:
            if pid in self.payloads and pid in self.vectors:
                p_load = self.payloads[pid]
                ret.append(
                    VectorPoint(
                        point_id=pid,
                        chunk_id=p_load["chunk_id"],
                        vector=self.vectors[pid].tolist(),
                        global_article_id=p_load["global_article_id"],
                        law_id=p_load["law_id"],
                        article_identifier=p_load["article_identifier"],
                        chunk_order=p_load["chunk_order"],
                        total_chunks=p_load["total_chunks"],
                        chunking_version=p_load["chunking_version"],
                        token_count=p_load["token_count"],
                        content_hash=p_load["content_hash"],
                        chunking_strategy=p_load["chunking_strategy"],
                        chunk_text=p_load["chunk_text"],
                        is_active=p_load.get("is_active", True),
                    )
                )
            else:
                ret.append(None)
        return ret

    def delete(self, point_ids: List[str]) -> int:
        """Xóa các điểm khỏi kho."""
        deleted = 0
        for pid in point_ids:
            if pid in self.payloads:
                del self.payloads[pid]
                del self.vectors[pid]
                deleted += 1

        if deleted > 0:
            self.point_ids = list(self.vectors.keys())
            self.index = faiss.IndexFlatIP(self.dimension)
            if self.point_ids:
                all_vecs = np.stack([self.vectors[pid] for pid in self.point_ids], axis=0)
                self.index.add(np.ascontiguousarray(all_vecs, dtype=np.float32))

        return deleted

    def count(self, filter_dict: Optional[Dict[str, Any]] = None) -> int:
        """Đếm số lượng điểm vector có trong kho."""
        if not filter_dict:
            return len(self.point_ids)
        cnt = 0
        for pid in self.point_ids:
            if self._matches_filter(self.payloads.get(pid, {}), filter_dict):
                cnt += 1
        return cnt

    def save(self, path: Union[str, Path]) -> None:
        """Lưu toàn bộ chỉ mục FAISS, ID và payload metadata xuống đĩa."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(save_dir / "index.faiss"))
        with open(save_dir / "payloads.json", "w", encoding="utf-8") as f:
            json.dump(self.payloads, f, indent=2, ensure_ascii=False)
        with open(save_dir / "point_ids.json", "w", encoding="utf-8") as f:
            json.dump(self.point_ids, f)

        if self.point_ids:
            all_vecs = np.stack([self.vectors[pid] for pid in self.point_ids], axis=0)
            np.save(save_dir / "vectors.npy", all_vecs)

    @classmethod
    def load(cls, path: Union[str, Path]) -> Self:
        """Nạp chỉ mục và khôi phục toàn vẹn trạng thái từ đĩa."""
        load_dir = Path(path)
        instance = cls()
        instance.index = faiss.read_index(str(load_dir / "index.faiss"))
        with open(load_dir / "payloads.json", "r", encoding="utf-8") as f:
            instance.payloads = json.load(f)
        with open(load_dir / "point_ids.json", "r", encoding="utf-8") as f:
            instance.point_ids = json.load(f)

        vecs_file = load_dir / "vectors.npy"
        if vecs_file.exists():
            vecs_mat = np.load(vecs_file)
            for idx, pid in enumerate(instance.point_ids):
                instance.vectors[pid] = vecs_mat[idx]

        return instance
