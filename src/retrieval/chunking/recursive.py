"""Chiến lược phân đoạn văn bản đệ quy theo đoạn văn, câu và từ."""

from typing import Any, List, Optional, Tuple
from src.retrieval.chunking.base import BaseChunker
from src.retrieval.chunking.models import (
    LegalChunk,
    compute_content_hash,
    generate_deterministic_chunk_id,
)


class RecursiveChunker(BaseChunker):
    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        target_size: int = 512,
        overlap: float = 0.0,
        chunking_version: Optional[str] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        if target_size <= 0:
            raise ValueError(f"target_size phải > 0, nhận được: {target_size}")
        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"overlap phải nằm trong [0.0, 1.0), nhận được: {overlap}")

        overlap_pct = int(overlap * 100)
        strategy_name = f"recursive_{target_size}_ov{overlap_pct}"
        version = chunking_version or f"recursive_{target_size}_o{overlap_pct}_v1"

        super().__init__(
            strategy_name=strategy_name,
            target_size=target_size,
            overlap=overlap,
            chunking_version=version,
            tokenizer=tokenizer,
        )
        self.fallback_count: int = 0

    def _split_into_atomic_units(
        self,
        text: str,
        start_offset: int,
        sep_idx: int = 0,
    ) -> List[Tuple[int, int]]:
        token_len = self.count_tokens(text)
        if token_len <= self.target_size or sep_idx >= len(self.SEPARATORS):
            if token_len > self.target_size:
                self.fallback_count += 1
                return self._token_fallback_spans(text, start_offset)
            return [(start_offset, start_offset + len(text))]

        sep = self.SEPARATORS[sep_idx]
        if sep == "":
            return self._token_fallback_spans(text, start_offset)

        if sep not in text:
            return self._split_into_atomic_units(text, start_offset, sep_idx + 1)

        raw_parts = text.split(sep)
        spans: List[Tuple[int, int]] = []
        curr_pos = 0

        for i, part in enumerate(raw_parts):
            if not part:
                curr_pos += len(sep) if i < len(raw_parts) - 1 else 0
                continue

            part_start = text.find(part, curr_pos)
            part_end = part_start + len(part)
            curr_pos = part_end

            part_token_len = self.count_tokens(part)
            if part_token_len <= self.target_size:
                spans.append((start_offset + part_start, start_offset + part_end))
            else:
                sub_spans = self._split_into_atomic_units(
                    part, start_offset + part_start, sep_idx + 1
                )
                spans.extend(sub_spans)

        return spans

    def _token_fallback_spans(
        self, text: str, start_offset: int
    ) -> List[Tuple[int, int]]:
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
            verbose=False,
        )
        offsets = encoding["offset_mapping"]
        total_tokens = len(offsets)

        spans: List[Tuple[int, int]] = []
        for i in range(0, total_tokens, self.target_size):
            end_i = min(i + self.target_size, total_tokens)
            c_s = offsets[i][0]
            c_e = offsets[end_i - 1][1]
            spans.append((start_offset + c_s, start_offset + c_e))
        return spans

    def chunk_article(
        self,
        text: str,
        global_article_id: int,
        law_id: str,
        article_identifier: str,
    ) -> List[LegalChunk]:
        if not text or not text.strip():
            return []

        total_tokens = self.count_tokens(text)
        if total_tokens <= self.target_size:
            c_hash = compute_content_hash(text)
            cid = generate_deterministic_chunk_id(
                global_article_id=global_article_id,
                chunking_version=self.chunking_version,
                chunk_order=0,
                content_hash=c_hash,
            )
            return [
                LegalChunk(
                    chunk_id=cid,
                    global_article_id=global_article_id,
                    law_id=law_id,
                    article_identifier=article_identifier,
                    chunk_order=0,
                    total_chunks=1,
                    chunk_text=text,
                    token_count=total_tokens,
                    structure_path="article",
                    content_hash=c_hash,
                    chunking_strategy=self.strategy_name,
                    chunking_version=self.chunking_version,
                    char_start=0,
                    char_end=len(text),
                    structural_unit="article",
                )
            ]

        atomic_spans = self._split_into_atomic_units(text, 0, 0)
        if not atomic_spans:
            return []

        merged_chunks_info: List[Tuple[int, int, str, int]] = []
        curr_span_start = atomic_spans[0][0]
        curr_span_end = atomic_spans[0][1]

        for next_s, next_e in atomic_spans[1:]:
            tentative_text = text[curr_span_start:next_e]
            tentative_toks = self.count_tokens(tentative_text)

            if tentative_toks <= self.target_size:
                curr_span_end = next_e
            else:
                chunk_content = text[curr_span_start:curr_span_end]
                tok_cnt = self.count_tokens(chunk_content)
                merged_chunks_info.append(
                    (curr_span_start, curr_span_end, chunk_content, tok_cnt)
                )
                curr_span_start = next_s
                curr_span_end = next_e

        last_chunk_content = text[curr_span_start:curr_span_end]
        last_tok_cnt = self.count_tokens(last_chunk_content)
        merged_chunks_info.append(
            (curr_span_start, curr_span_end, last_chunk_content, last_tok_cnt)
        )

        total_chunks = len(merged_chunks_info)
        result_chunks: List[LegalChunk] = []

        for order, (c_s, c_e, c_text, c_toks) in enumerate(merged_chunks_info):
            c_hash = compute_content_hash(c_text)
            cid = generate_deterministic_chunk_id(
                global_article_id=global_article_id,
                chunking_version=self.chunking_version,
                chunk_order=order,
                content_hash=c_hash,
            )
            result_chunks.append(
                LegalChunk(
                    chunk_id=cid,
                    global_article_id=global_article_id,
                    law_id=law_id,
                    article_identifier=article_identifier,
                    chunk_order=order,
                    total_chunks=total_chunks,
                    chunk_text=c_text,
                    token_count=c_toks,
                    structure_path=f"article/recursive_{order}",
                    content_hash=c_hash,
                    chunking_strategy=self.strategy_name,
                    chunking_version=self.chunking_version,
                    char_start=c_s,
                    char_end=c_e,
                    structural_unit="recursive_unit",
                )
            )

        return result_chunks
