# src/retrieval/chunking/structure_aware.py
"""Chiến lược phân đoạn nhận biết cấu trúc pháp lý (Structure-aware Legal Chunking) cho Phase 7.

Tuân thủ Mục 7.5 và Mục 9.3 của Đặc tả Phase 7:
    - Phân cấp cấu trúc pháp lý chuẩn Việt Nam: Điều -> Khoản -> Điểm -> Câu -> Token fallback.
    - Phát hiện tất định ranh giới Khoản (1., 2., Khoản 1,...) và Điểm (a), b), Điểm a,...).
    - Giữ nguyên vẹn điều luật nếu vừa vặn trong target_size.
    - Ghi nhận structure_path, parent_clause_id, parent_point_id và fallback_count.
"""

import re
from typing import Any, List, Optional, Tuple
from src.retrieval.chunking.base import BaseChunker
from src.retrieval.chunking.models import (
    LegalChunk,
    compute_content_hash,
    generate_deterministic_chunk_id,
)

# Pattern nhận diện Khoản: \n1. hoặc ^1. hoặc Khoản 1. hoặc \t1.
CLAUSE_PATTERN = re.compile(
    r"(?:(?<=\n)|(?<=\A)|(?<=\t)|(?<=\r))(?:\d+\.|\bKhoản\s+\d+[\.:])\s+",
    re.UNICODE,
)

# Pattern nhận diện Điểm: a), b), c), đ) hoặc Điểm a)
POINT_PATTERN = re.compile(
    r"(?:(?<=\n)|(?<=\A)|(?<=\t)|(?<=\r)|(?<=\s))(?:\b[a-zđ]\)|\bĐiểm\s+[a-zđ][\.:\)])\s+",
    re.UNICODE,
)


class StructureAwareChunker(BaseChunker):
    """Bộ phân đoạn thông minh dựa trên phân cấp cấu trúc văn bản pháp luật."""

    def __init__(
        self,
        target_size: int = 512,
        clause_first_only: bool = False,
        overlap: float = 0.0,
        chunking_version: Optional[str] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        if target_size <= 0:
            raise ValueError(f"target_size phải > 0, nhận được: {target_size}")
        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"overlap phải nằm trong [0.0, 1.0), nhận được: {overlap}")

        overlap_pct = int(overlap * 100)
        self.clause_first_only = clause_first_only

        if clause_first_only:
            strategy_name = f"structure_aware_clause_first_{target_size}_ov{overlap_pct}"
            version = chunking_version or f"struct_clause_{target_size}_o{overlap_pct}_v1"
        else:
            strategy_name = f"structure_aware_hierarchical_{target_size}_ov{overlap_pct}"
            version = chunking_version or f"struct_hier_{target_size}_o{overlap_pct}_v1"

        super().__init__(
            strategy_name=strategy_name,
            target_size=target_size,
            overlap=overlap,
            chunking_version=version,
            tokenizer=tokenizer,
        )

        self.clause_detected_articles: int = 0
        self.point_detected_articles: int = 0
        self.sentence_fallback_count: int = 0
        self.token_fallback_count: int = 0

    def _extract_clause_spans(self, text: str) -> List[Tuple[int, int, str]]:
        """Phát hiện và trích xuất các Khoản trong văn bản điều luật."""
        matches = list(CLAUSE_PATTERN.finditer(text))
        if not matches:
            return []

        spans: List[Tuple[int, int, str]] = []
        for i, m in enumerate(matches):
            c_start = m.start()
            c_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            raw_label = m.group().strip()
            if raw_label.endswith("."):
                clause_num = raw_label[:-1].strip()
                if clause_num.isdigit():
                    clause_label = f"Khoản {clause_num}"
                else:
                    clause_label = raw_label
            else:
                clause_label = raw_label
            spans.append((c_start, c_end, clause_label))

        # Nếu có phần mở đầu trước Khoản 1 (lời mở đầu điều luật)
        if matches[0].start() > 0:
            preamble_text = text[: matches[0].start()].strip()
            if preamble_text:
                spans.insert(0, (0, matches[0].start(), "Lời mở đầu"))

        return spans

    def _extract_point_spans(self, text: str, offset: int) -> List[Tuple[int, int, str]]:
        """Phát hiện và trích xuất các Điểm trong một Khoản."""
        matches = list(POINT_PATTERN.finditer(text))
        if not matches:
            return []

        spans: List[Tuple[int, int, str]] = []
        for i, m in enumerate(matches):
            p_start = m.start()
            p_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            raw_p = m.group().strip()
            if raw_p.endswith(")"):
                pt_char = raw_p[:-1].strip()
                point_label = f"Điểm {pt_char}"
            else:
                point_label = raw_p
            spans.append((offset + p_start, offset + p_end, point_label))

        if matches[0].start() > 0:
            p_preamble = text[: matches[0].start()].strip()
            if p_preamble:
                spans.insert(0, (offset, offset + matches[0].start(), "Phần dẫn"))

        return spans

    def _token_fallback(self, text: str, abs_start: int) -> List[Tuple[int, int, str, str]]:
        """Token fallback cắt lát theo token cho đơn vị vẫn vượt ngưỡng."""
        self.token_fallback_count += 1
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
            verbose=False,
        )
        offsets = encoding["offset_mapping"]
        total_tokens = len(offsets)

        units: List[Tuple[int, int, str, str]] = []
        for i in range(0, total_tokens, self.target_size):
            end_i = min(i + self.target_size, total_tokens)
            c_s = offsets[i][0]
            c_e = offsets[end_i - 1][1]
            units.append((abs_start + c_s, abs_start + c_e, "token_fallback", "token"))
        return units

    def chunk_article(
        self,
        text: str,
        global_article_id: int,
        law_id: str,
        article_identifier: str,
    ) -> List[LegalChunk]:
        """Phân đoạn điều luật theo phân cấp cấu trúc pháp lý."""
        if not text or not text.strip():
            return []

        total_tokens = self.count_tokens(text)

        # 1. Điều luật vừa vặn target_size -> 1 chunk duy nhất
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
                    structure_path="Điều",
                    content_hash=c_hash,
                    chunking_strategy=self.strategy_name,
                    chunking_version=self.chunking_version,
                    char_start=0,
                    char_end=len(text),
                    structural_unit="article",
                    structural_label="Điều",
                )
            ]

        # 2. Thử trích xuất các Khoản
        clause_spans = self._extract_clause_spans(text)
        units: List[Tuple[int, int, str, str]] = []  # (start, end, path, unit_type)

        if clause_spans:
            self.clause_detected_articles += 1
            for c_start, c_end, c_label in clause_spans:
                clause_text = text[c_start:c_end]
                clause_tokens = self.count_tokens(clause_text)

                if clause_tokens <= self.target_size or self.clause_first_only:
                    # Khoản vừa vặn hoặc ở chế độ clause-first
                    units.append((c_start, c_end, f"Điều > {c_label}", "clause"))
                else:
                    # Khoản vượt ngưỡng -> thử trích xuất Điểm
                    point_spans = self._extract_point_spans(clause_text, c_start)
                    if point_spans:
                        self.point_detected_articles += 1
                        for p_start, p_end, p_label in point_spans:
                            point_text = text[p_start:p_end]
                            point_tokens = self.count_tokens(point_text)

                            if point_tokens <= self.target_size:
                                units.append(
                                    (
                                        p_start,
                                        p_end,
                                        f"Điều > {c_label} > {p_label}",
                                        "point",
                                    )
                                )
                            else:
                                # Điểm vượt ngưỡng -> token fallback
                                units.extend(
                                    self._token_fallback(point_text, p_start)
                                )
                    else:
                        # Không có Điểm -> token fallback
                        units.extend(self._token_fallback(clause_text, c_start))
        else:
            # Không phát hiện cấu trúc Khoản -> token fallback
            units.extend(self._token_fallback(text, 0))

        # 3. Gom nhóm các đơn vị nhỏ liền kề nếu không vượt target_size (tránh phân mảnh quá mức)
        merged_units: List[Tuple[int, int, str, str, str, int]] = []
        if units:
            curr_s, curr_e, curr_path, curr_type = units[0]
            for next_s, next_e, next_path, next_type in units[1:]:
                tentative_text = text[curr_s:next_e]
                tentative_toks = self.count_tokens(tentative_text)
                if tentative_toks <= self.target_size and not self.clause_first_only:
                    curr_e = next_e
                    curr_path = f"{curr_path} + {next_path}"
                    curr_type = "merged_legal_units"
                else:
                    c_str = text[curr_s:curr_e]
                    merged_units.append(
                        (curr_s, curr_e, curr_path, curr_type, c_str, self.count_tokens(c_str))
                    )
                    curr_s, curr_e, curr_path, curr_type = next_s, next_e, next_path, next_type

            last_str = text[curr_s:curr_e]
            merged_units.append(
                (curr_s, curr_e, curr_path, curr_type, last_str, self.count_tokens(last_str))
            )

        total_chunks = len(merged_units)
        result_chunks: List[LegalChunk] = []

        for order, (c_s, c_e, c_path, c_type, c_text, c_toks) in enumerate(merged_units):
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
                    structure_path=c_path,
                    content_hash=c_hash,
                    chunking_strategy=self.strategy_name,
                    chunking_version=self.chunking_version,
                    char_start=c_s,
                    char_end=c_e,
                    structural_unit=c_type,
                    structural_label=c_path,
                )
            )

        return result_chunks
