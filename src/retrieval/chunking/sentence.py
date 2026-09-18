# src/retrieval/chunking/sentence.py
"""Chiến lược phân đoạn theo ranh giới câu (Sentence-based Chunking) cho Phase 7.

Tuân thủ Mục 7.3 của Đặc tả Phase 7:
    - Phân đoạn câu tất định, bảo toàn ranh giới câu tiếng Việt.
    - Không làm mất ký tự nguồn, xử lý các trường hợp ngoại lệ viết tắt/số hiệu pháp lý.
    - Cơ chế gom nhóm câu liên tiếp đạt ngân sách target_size và fallback khi câu quá dài.
"""

import re
from typing import Any, List, Optional, Tuple
from src.retrieval.chunking.base import BaseChunker
from src.retrieval.chunking.models import (
    LegalChunk,
    compute_content_hash,
    generate_deterministic_chunk_id,
)

# Pattern nhận diện kết thúc câu trong văn bản pháp luật tiếng Việt
SENTENCE_SPLIT_PATTERN = re.compile(
    r"(?<=[.!?])\s+(?=[A-ZĐÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ0-9])"
)
# Các trường hợp viết tắt không phải kết thúc câu
ABBREVIATION_PATTERN = re.compile(
    r"\b(?:Điều|Khoản|Điểm|số|Số|ngày|NĐ-CP|TT-BTC|NQ-CP|QĐ-TTg|TP|TW|K/c)\.\s*$",
    re.IGNORECASE,
)


class SentenceChunker(BaseChunker):
    """Bộ phân đoạn văn bản dựa trên ghép các câu hoàn chỉnh."""

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
        strategy_name = f"sentence_{target_size}_ov{overlap_pct}"
        version = chunking_version or f"sentence_{target_size}_o{overlap_pct}_v1"

        super().__init__(
            strategy_name=strategy_name,
            target_size=target_size,
            overlap=overlap,
            chunking_version=version,
            tokenizer=tokenizer,
        )
        self.fallback_count: int = 0

    def _split_into_sentences(self, text: str) -> List[Tuple[int, int]]:
        """Phân tách văn bản thành các khoảng ký tự (start, end) của câu."""
        if not text:
            return []

        # Tách trước theo dòng để bảo toàn cấu trúc dòng
        lines_spans: List[Tuple[int, int]] = []
        curr = 0
        for line in text.split("\n"):
            line_len = len(line)
            if line_len > 0:
                lines_spans.append((curr, curr + line_len))
            curr += line_len + 1  # 1 cho ký tự '\n'

        sentence_spans: List[Tuple[int, int]] = []
        for l_start, l_end in lines_spans:
            line_text = text[l_start:l_end]
            if not line_text.strip():
                continue

            # Phân tách trong từng dòng
            matches = list(SENTENCE_SPLIT_PATTERN.finditer(line_text))
            if not matches:
                sentence_spans.append((l_start, l_end))
                continue

            last_idx = 0
            for m in matches:
                split_point = m.start() + 1
                segment = line_text[last_idx:split_point]
                # Kiểm tra ngoại lệ viết tắt
                if ABBREVIATION_PATTERN.search(segment):
                    continue
                sentence_spans.append((l_start + last_idx, l_start + split_point))
                last_idx = m.end()

            if last_idx < len(line_text):
                sentence_spans.append((l_start + last_idx, l_end))

        return sentence_spans

    def chunk_article(
        self,
        text: str,
        global_article_id: int,
        law_id: str,
        article_identifier: str,
    ) -> List[LegalChunk]:
        """Phân đoạn điều luật theo từng câu hoàn chỉnh."""
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

        raw_sent_spans = self._split_into_sentences(text)
        if not raw_sent_spans:
            raw_sent_spans = [(0, len(text))]

        # Chuẩn hóa các câu: nếu câu nào quá dài thì token fallback
        atomic_spans: List[Tuple[int, int]] = []
        for s_start, s_end in raw_sent_spans:
            sent_str = text[s_start:s_end]
            tok_len = self.count_tokens(sent_str)
            if tok_len <= self.target_size:
                atomic_spans.append((s_start, s_end))
            else:
                self.fallback_count += 1
                # Fallback cắt lát theo token cho câu ngoại cỡ
                encoding = self.tokenizer(
                    sent_str,
                    add_special_tokens=False,
                    truncation=False,
                    return_offsets_mapping=True,
                    verbose=False,
                )
                offsets = encoding["offset_mapping"]
                for i in range(0, len(offsets), self.target_size):
                    end_i = min(i + self.target_size, len(offsets))
                    atomic_spans.append(
                        (s_start + offsets[i][0], s_start + offsets[end_i - 1][1])
                    )

        # Gom nhóm các câu liên tiếp theo target_size
        merged_chunks_info: List[Tuple[int, int, str, int]] = []
        curr_start = atomic_spans[0][0]
        curr_end = atomic_spans[0][1]

        for next_s, next_e in atomic_spans[1:]:
            tentative_text = text[curr_start:next_e]
            tok_cnt = self.count_tokens(tentative_text)
            if tok_cnt <= self.target_size:
                curr_end = next_e
            else:
                chunk_str = text[curr_start:curr_end]
                merged_chunks_info.append(
                    (curr_start, curr_end, chunk_str, self.count_tokens(chunk_str))
                )
                curr_start = next_s
                curr_end = next_e

        last_str = text[curr_start:curr_end]
        merged_chunks_info.append(
            (curr_start, curr_end, last_str, self.count_tokens(last_str))
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
                    structure_path=f"article/sentence_{order}",
                    content_hash=c_hash,
                    chunking_strategy=self.strategy_name,
                    chunking_version=self.chunking_version,
                    char_start=c_s,
                    char_end=c_e,
                    structural_unit="sentence_group",
                )
            )

        return result_chunks
