# src/retrieval/chunking/fixed_size.py
"""Chiến lược phân đoạn kích thước cố định (Fixed-size Chunking) cho Phase 7.

Tuân thủ Mục 7.1 của Đặc tả Phase 7:
    - Ứng viên kích thước bắt buộc: 256, 512, 1024 tokens.
    - Hỗ trợ overlap: 0%, 10%, 20%.
    - Bảo toàn 100% text gốc, theo dõi chính xác char_start và char_end.
    - Xử lý điều luật rỗng: trả về [] (total_chunks = 0).
"""

from typing import Any, List, Optional
from src.retrieval.chunking.base import BaseChunker
from src.retrieval.chunking.models import (
    LegalChunk,
    compute_content_hash,
    generate_deterministic_chunk_id,
)


class FixedSizeChunker(BaseChunker):
    """Bộ phân đoạn văn bản dựa trên số lượng token cố định."""

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
        strategy_name = f"fixed_size_{target_size}_ov{overlap_pct}"
        version = chunking_version or f"fixed_{target_size}_o{overlap_pct}_v1"

        super().__init__(
            strategy_name=strategy_name,
            target_size=target_size,
            overlap=overlap,
            chunking_version=version,
            tokenizer=tokenizer,
        )

    def chunk_article(
        self,
        text: str,
        global_article_id: int,
        law_id: str,
        article_identifier: str,
    ) -> List[LegalChunk]:
        """Phân đoạn điều luật theo cửa sổ token cố định trượt."""
        if not text or not text.strip():
            return []

        # Tokenize có trích xuất ánh xạ offset ký tự
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
            verbose=False,
        )
        input_ids = encoding["input_ids"]
        offset_mapping = encoding["offset_mapping"]
        total_tokens = len(input_ids)

        if total_tokens == 0:
            return []

        # Nếu toàn bộ điều luật nhỏ hơn hoặc bằng target_size: tạo 1 chunk duy nhất
        if total_tokens <= self.target_size:
            c_hash = compute_content_hash(text)
            cid = generate_deterministic_chunk_id(
                global_article_id=global_article_id,
                chunking_version=self.chunking_version,
                chunk_order=0,
                content_hash=c_hash,
            )
            chunk = LegalChunk(
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
            return [chunk]

        # Tính bước nhảy cửa sổ trượt
        step = max(1, int(self.target_size * (1.0 - self.overlap)))
        raw_chunks = []

        start_tok = 0
        while start_tok < total_tokens:
            end_tok = min(start_tok + self.target_size, total_tokens)

            # Lấy vị trí ký tự đầu và cuối
            char_s = offset_mapping[start_tok][0]
            char_e = offset_mapping[end_tok - 1][1]

            chunk_content = text[char_s:char_e]
            tok_count = end_tok - start_tok

            raw_chunks.append((chunk_content, char_s, char_e, tok_count))

            if end_tok >= total_tokens:
                break
            start_tok += step

        total_chunks = len(raw_chunks)
        result_chunks: List[LegalChunk] = []

        for order, (c_text, c_s, c_e, c_tokens) in enumerate(raw_chunks):
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
                    token_count=c_tokens,
                    structure_path=f"article/fixed_{order}",
                    content_hash=c_hash,
                    chunking_strategy=self.strategy_name,
                    chunking_version=self.chunking_version,
                    char_start=c_s,
                    char_end=c_e,
                    structural_unit="fixed_token_window",
                )
            )

        return result_chunks
