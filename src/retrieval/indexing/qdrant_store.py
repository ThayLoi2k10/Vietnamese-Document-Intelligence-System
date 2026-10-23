"""Kho lưu trữ vector Qdrant cục bộ hỗ trợ pre-filtering và lưu trữ bền vững."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Union
import numpy as np
from qdrant_client import QdrantClient, models

from src.retrieval.indexing.base import BaseVectorStore
from src.retrieval.indexing.models import VectorPoint, VectorSearchResult


class QdrantVectorStore(BaseVectorStore):
    def __init__(
        self,
        storage_path: Union[str, Path],
        dimension: int = 1024,
        collection_name: str = "legal_chunks",
    ) -> None:
        super().__init__(dimension=dimension, collection_name=collection_name)
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.client = QdrantClient(path=str(self.storage_path))
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="law_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="global_article_id",
                field_schema=models.PayloadSchemaType.INTEGER,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="chunking_version",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="is_active",
                field_schema=models.PayloadSchemaType.BOOL,
            )

    def _build_filter(self, filter_dict: Optional[Dict[str, Any]]) -> Optional[models.Filter]:
        if not filter_dict:
            return None
        must_conditions = [
            models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in filter_dict.items()
        ]
        return models.Filter(must=must_conditions)

    def upsert(self, points: List[VectorPoint], batch_size: int = 256) -> int:
        if not points:
            return 0

        total_upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            qdrant_points = [
                models.PointStruct(
                    id=p.point_id,
                    vector=p.vector if isinstance(p.vector, list) else p.vector.tolist(),
                    payload=p.to_payload(),
                )
                for p in batch
            ]
            self.client.upsert(
                collection_name=self.collection_name,
                points=qdrant_points,
                wait=True,
            )
            total_upserted += len(batch)

        return total_upserted

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 100,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        q_vec = query_vector.flatten().tolist()
        q_filter = self._build_filter(filter_dict)

        res = self.client.query_points(
            collection_name=self.collection_name,
            query=q_vec,
            query_filter=q_filter,
            limit=limit,
            with_payload=True,
        )

        results: List[VectorSearchResult] = []
        for rank_i, hit in enumerate(res.points, start=1):
            payload = hit.payload or {}
            results.append(
                VectorSearchResult(
                    point_id=str(hit.id),
                    chunk_id=str(payload.get("chunk_id", hit.id)),
                    global_article_id=int(payload.get("global_article_id", -1)),
                    score=float(hit.score),
                    rank=rank_i,
                    payload=payload,
                )
            )

        return results

    def get(self, point_ids: List[str]) -> List[Optional[VectorPoint]]:
        if not point_ids:
            return []
        res = self.client.retrieve(
            collection_name=self.collection_name,
            ids=point_ids,
            with_vectors=True,
            with_payload=True,
        )
        hit_map = {str(hit.id): hit for hit in res}

        ret: List[Optional[VectorPoint]] = []
        for pid in point_ids:
            hit = hit_map.get(str(pid))
            if hit:
                p_load = hit.payload or {}
                ret.append(
                    VectorPoint(
                        point_id=str(hit.id),
                        chunk_id=str(p_load.get("chunk_id", hit.id)),
                        vector=hit.vector if isinstance(hit.vector, list) else [],
                        global_article_id=int(p_load.get("global_article_id", -1)),
                        law_id=str(p_load.get("law_id", "")),
                        article_identifier=str(p_load.get("article_identifier", "")),
                        chunk_order=int(p_load.get("chunk_order", 0)),
                        total_chunks=int(p_load.get("total_chunks", 1)),
                        chunking_version=str(p_load.get("chunking_version", "")),
                        token_count=int(p_load.get("token_count", 0)),
                        content_hash=str(p_load.get("content_hash", "")),
                        chunking_strategy=str(p_load.get("chunking_strategy", "")),
                        chunk_text=str(p_load.get("chunk_text", "")),
                        is_active=bool(p_load.get("is_active", True)),
                        char_start=int(p_load.get("char_start", 0)),
                        char_end=int(p_load.get("char_end", 0)),
                    )
                )
            else:
                ret.append(None)
        return ret

    def delete(self, point_ids: List[str]) -> int:
        if not point_ids:
            return 0
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=point_ids),
            wait=True,
        )
        return len(point_ids)

    def count(self, filter_dict: Optional[Dict[str, Any]] = None) -> int:
        q_filter = self._build_filter(filter_dict)
        res = self.client.count(
            collection_name=self.collection_name,
            count_filter=q_filter,
            exact=True,
        )
        return res.count

    def save(self, path: Union[str, Path]) -> None:
        pass

    @classmethod
    def load(cls, path: Union[str, Path]) -> Self:
        return cls(storage_path=path)
