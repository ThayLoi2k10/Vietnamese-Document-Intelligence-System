"""Lớp cơ sở trừu tượng cho các chiến lược phân đoạn văn bản pháp luật."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional
import pandas as pd
from transformers import AutoTokenizer

from src.retrieval.chunking.models import LegalChunk

OFFICIAL_MODEL_NAME = "BAAI/bge-m3"


class BaseChunker(ABC):
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
        pass

    def chunk_corpus(
        self, corpus_df: pd.DataFrame, show_progress: bool = False
    ) -> List[LegalChunk]:
        all_chunks: List[LegalChunk] = []

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
