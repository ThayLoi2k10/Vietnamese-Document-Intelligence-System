# src/retrieval/chunking/base.py
"""Lớp trừu tượng BaseChunker cho toàn bộ các chiến lược phân đoạn trong Phase 7.

Tuân thủ nghiêm ngặt đặc tả Phase 7:
    - Mục 6: Core Design Principle (cân bằng độ đặc thù ngữ nghĩa và tính toàn vẹn pháp lý)
    - Mục 12: Provenance Invariants (bảo toàn điều luật rỗng, không làm mất text)
    - Mục 13: Deterministic Versioning & Identity
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd
from transformers import AutoTokenizer

from src.retrieval.chunking.models import (
    LegalChunk,
    compute_content_hash,
    generate_deterministic_chunk_id,
)

OFFICIAL_MODEL_NAME = "BAAI/bge-m3"


class BaseChunker(ABC):
    """Lớp cơ sở trừu tượng cho tất cả các thuật toán chunking."""

    def __init__(
        self,
        strategy_name: str,
        target_size: int = 512,
        overlap: float = 0.0,
        chunking_version: str = "v1.0",
        tokenizer: Optional[Any] = None,
    ) -> None:
        self.strategy_name = strategy_name
        self.target_size = target_size
        self.overlap = overlap
        self.chunking_version = chunking_version

        if tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    OFFICIAL_MODEL_NAME, local_files_only=True
                )
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(OFFICIAL_MODEL_NAME)
        else:
            self.tokenizer = tokenizer

    def count_tokens(self, text: str) -> int:
        """Đếm chính xác số lượng token của văn bản theo BGE-M3 tokenizer."""
        if not text:
            return 0
        ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )
        return len(ids)

    @abstractmethod
    def chunk_article(
        self,
        text: str,
        global_article_id: int,
        law_id: str,
        article_identifier: str,
    ) -> List[LegalChunk]:
        """Phân đoạn một điều luật thành danh sách LegalChunk.
        
        Nếu điều luật rỗng, bắt buộc trả về danh sách rỗng [] (total_chunks = 0).
        """
        pass

    def chunk_corpus(
        self, corpus_df: pd.DataFrame, show_progress: bool = False
    ) -> List[LegalChunk]:
        """Phân đoạn toàn bộ dataframe corpus điều luật thành danh sách phẳng LegalChunk."""
        all_chunks: List[LegalChunk] = []

        # Đảm bảo các cột định danh cần thiết
        id_col = "global_article_id" if "global_article_id" in corpus_df.columns else "article_id"
        text_col = "text" if "text" in corpus_df.columns else "text_preprocessed"
        law_col = "law_id" if "law_id" in corpus_df.columns else "doc_id"
        aid_col = "aid" if "aid" in corpus_df.columns else id_col

        for _, row in corpus_df.iterrows():
            gid = int(row[id_col])
            raw_text = row[text_col]
            text = "" if (raw_text is None or pd.isna(raw_text)) else str(raw_text)
            lid = str(row.get(law_col, ""))
            aid = str(row.get(aid_col, str(gid)))

            chunks = self.chunk_article(
                text=text,
                global_article_id=gid,
                law_id=lid,
                article_identifier=aid,
            )
            all_chunks.extend(chunks)

        return all_chunks
