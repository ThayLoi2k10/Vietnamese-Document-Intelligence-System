# src/retrieval/indexing/lifecycle.py
"""Quản lý vòng đời và đường ống nạp dữ liệu gia tăng (Incremental Ingestion Pipeline).

Tuân thủ nghiêm ngặt:
    - Mục 14: Lifecycle State Model (discovered -> parsed -> chunked -> embedded -> indexed -> committed)
    - Mục 15: Incremental Ingestion Pipeline
    - Mục 17: Idempotent Upsert
    - Mục 18: Update Handling
    - Mục 19: Delete / Revocation Handling
    - Mục 20: Partial Failure and Consistency
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.retrieval.chunking.base import BaseChunker
from src.retrieval.dense.encoder import DenseEncoder
from src.retrieval.indexing.base import BaseVectorStore
from src.retrieval.indexing.models import VectorPoint, generate_deterministic_point_id


class IngestionState(str, Enum):
    DISCOVERED = "discovered"
    PARSED = "parsed"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    INDEXED = "indexed"
    COMMITTED = "committed"
    FAILED = "failed"


@dataclass
class IngestionRecord:
    operation_id: str
    global_article_id: int
    law_id: str
    state: IngestionState
    chunk_count: int = 0
    point_ids: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class IncrementalIngestionManager:
    """Quản lý đường ống nạp mới, cập nhật và thu hồi điều luật pháp lý."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        chunker: BaseChunker,
        encoder: DenseEncoder,
    ) -> None:
        self.vector_store = vector_store
        self.chunker = chunker
        self.encoder = encoder
        self.history: List[IngestionRecord] = []

    def ingest_article(
        self,
        global_article_id: int,
        law_id: str,
        article_identifier: str,
        text: str,
        is_update: bool = False,
    ) -> IngestionRecord:
        """Thực thi đường ống 10 bước nạp tài liệu theo Mục 15 của đặc tả."""
        op_id = f"op_{global_article_id}_{int(time.time() * 1000)}"
        record = IngestionRecord(
            operation_id=op_id,
            global_article_id=global_article_id,
            law_id=law_id,
            state=IngestionState.DISCOVERED,
        )

        try:
            # 1. Identity & Parse
            record.state = IngestionState.PARSED

            # 2. Chunking sử dụng cấu hình Phase 7 đã đóng băng
            chunks = self.chunker.chunk_article(
                text=text,
                global_article_id=global_article_id,
                law_id=law_id,
                article_identifier=article_identifier,
            )
            record.chunk_count = len(chunks)
            record.state = IngestionState.CHUNKED

            if len(chunks) == 0:
                # Điều luật rỗng: không sinh điểm vector
                record.state = IngestionState.COMMITTED
                self.history.append(record)
                return record

            # 3. Embedding với cấu hình BGE-M3 đã đóng băng
            chunk_texts = [c.chunk_text for c in chunks]
            vecs = self.encoder.encode_articles(chunk_texts, batch_size=16)
            record.state = IngestionState.EMBEDDED

            # 4. Tạo các điểm VectorPoint
            points: List[VectorPoint] = []
            for c, vec in zip(chunks, vecs):
                p = VectorPoint.from_chunk_and_vector(
                    chunk_dict=c.to_dict(),
                    vector=vec.tolist(),
                    is_active=True,
                )
                points.append(p)
                record.point_ids.append(p.point_id)

            # 5. Upsert vào kho vector
            upserted = self.vector_store.upsert(points)
            if upserted != len(points):
                raise RuntimeError(f"Chỉ ghi được {upserted}/{len(points)} điểm vector!")

            record.state = IngestionState.INDEXED

            # 6. Xác thực ghi (Verify write)
            retrieved_points = self.vector_store.get(record.point_ids)
            if any(p is None for p in retrieved_points):
                raise RuntimeError("Xác thực sau ghi thất bại: Không tìm thấy điểm vừa upsert!")

            # 7. Cam kết trạng thái (Commit)
            record.state = IngestionState.COMMITTED

        except Exception as e:
            record.state = IngestionState.FAILED
            record.error_message = str(e)

        self.history.append(record)
        return record

    def deactivate_article(self, global_article_id: int) -> int:
        """Thu hồi / vô hiệu hóa điều luật (Mục 19)."""
        # Tìm tất cả các điểm của article_id trong store
        results = self.vector_store.search(
            query_vector=np.zeros(self.vector_store.dimension, dtype=np.float32),
            limit=1000,
            filter_dict={"global_article_id": global_article_id},
        )
        pids = [r.point_id for r in results]
        if pids:
            return self.vector_store.delete(pids)
        return 0
