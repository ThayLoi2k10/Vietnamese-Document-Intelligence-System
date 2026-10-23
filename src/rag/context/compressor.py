"""Passage compression and extractive focus engine for Vietnamese statutory articles."""

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import unicodedata


@dataclass
class CompressedPassage:
    """Outcome of legal passage compression focusing on query-relevant clauses."""

    original_text: str
    compressed_text: str
    original_length: int
    compressed_length: int
    compression_ratio: float  # compressed_length / original_length (<= 1.0)
    clauses_retained: List[str] = field(default_factory=list)
    is_compressed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LegalPassageCompressor:
    """Extractively compresses long statutory articles to retain query-relevant clauses."""

    CLAUSE_SPLIT_PATTERN = re.compile(r"(?:\n\s*(?=\d+\.\s+))")
    POINT_SPLIT_PATTERN = re.compile(r"(?:\n\s*(?=[a-zđ]\)\s+))", re.IGNORECASE)

    VIETNAMESE_STOPWORDS = {
        "và", "hoặc", "của", "cho", "các", "những", "được", "trong", "khi",
        "thì", "đó", "này", "với", "tại", "theo", "về", "có", "là", "một",
        "hai", "ba", "bốn", "năm", "ra", "vào", "đến", "từ", "lại", "đã",
        "sẽ", "đang", "như", "nào", "gì", "ai", "sao", "bao", "nhiêu",
        "quy", "định", "khoản", "điều", "văn", "bản", "luật", "nghị", "định",
    }

    def __init__(
        self,
        max_chars_per_article: int = 1500,
        min_chars_to_compress: int = 1000,
    ) -> None:
        self.max_chars_per_article = max_chars_per_article
        self.min_chars_to_compress = min_chars_to_compress

    @staticmethod
    def _normalize_word(word: str) -> str:
        word = unicodedata.normalize("NFKC", word)
        return re.sub(r"[^\w\s]", "", word).strip().lower()

    def _extract_query_keywords(self, query: str) -> Set[str]:
        words = re.findall(r"\w+", query.lower())
        keywords: Set[str] = set()
        for w in words:
            cleaned = self._normalize_word(w)
            if cleaned and len(cleaned) >= 2 and cleaned not in self.VIETNAMESE_STOPWORDS:
                keywords.add(cleaned)
        return keywords

    def _split_into_clauses(self, article_text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """Split article text into header and list of (clause_label, clause_text)."""
        stripped = article_text.strip()
        parts = self.CLAUSE_SPLIT_PATTERN.split(stripped)

        if len(parts) <= 1:
            # Không phát hiện cấu trúc khoản 1. 2. -> Thử chia theo đoạn văn \n\n
            paras = [p.strip() for p in re.split(r"\n\s*\n+", stripped) if p.strip()]
            if len(paras) <= 1:
                return stripped, []
            header = paras[0]
            clause_list = [(f"Đoạn {idx}", p) for idx, p in enumerate(paras[1:], start=1)]
            return header, clause_list

        header = parts[0].strip()
        clause_list: List[Tuple[str, str]] = []
        for p in parts[1:]:
            p_str = p.strip()
            if not p_str:
                continue
            # Lấy số khoản
            m = re.match(r"^(\d+)\.\s*", p_str)
            label = f"Khoản {m.group(1)}" if m else "Khoản"
            clause_list.append((label, p_str))

        return header, clause_list

    def _score_clause(self, clause_text: str, query_keywords: Set[str], query_raw: str) -> float:
        """Compute relevance score of a clause with respect to the query."""
        norm_text = self._normalize_word(clause_text)
        if not norm_text:
            return 0.0

        score = 0.0
        # 1. Khớp từ khóa
        clause_words = set(re.findall(r"\w+", norm_text))
        matched_words = query_keywords.intersection(clause_words)
        score += len(matched_words) * 2.0

        # 2. Khớp cụm từ (Bigram hoặc toàn bộ câu truy vấn nếu xuất hiện)
        norm_raw_query = self._normalize_word(query_raw)
        if norm_raw_query and len(norm_raw_query) > 5 and norm_raw_query in norm_text:
            score += 10.0

        # Khớp các cụm từ quan trọng 2-3 chữ
        q_tokens = list(query_keywords)
        for i in range(len(q_tokens) - 1):
            phrase = f"{q_tokens[i]} {q_tokens[i+1]}"
            if phrase in norm_text:
                score += 3.0

        return score

    def compress_article(
        self,
        query: str,
        article_text: str,
        max_chars: Optional[int] = None,
    ) -> CompressedPassage:
        """Compress a single article text by keeping the title and most relevant clauses."""
        if not article_text or not article_text.strip():
            return CompressedPassage(
                original_text="",
                compressed_text="",
                original_length=0,
                compressed_length=0,
                compression_ratio=1.0,
                clauses_retained=[],
                is_compressed=False,
            )

        orig_len = len(article_text)
        budget = max_chars or self.max_chars_per_article

        # Nếu độ dài bài viết vốn đã ngắn hơn ngưỡng nén -> giữ nguyên
        if orig_len <= self.min_chars_to_compress:
            return CompressedPassage(
                original_text=article_text,
                compressed_text=article_text,
                original_length=orig_len,
                compressed_length=orig_len,
                compression_ratio=1.0,
                clauses_retained=["toàn_văn"],
                is_compressed=False,
            )

        header, clauses = self._split_into_clauses(article_text)
        if not clauses:
            # Không tách được khoản -> cắt gọt an toàn theo độ dài
            cutoff_text = (
                article_text[:budget].strip() + "\n... [phần còn lại của điều luật dài được lược bớt] ..."
            )
            return CompressedPassage(
                original_text=article_text,
                compressed_text=cutoff_text,
                original_length=orig_len,
                compressed_length=len(cutoff_text),
                compression_ratio=round(len(cutoff_text) / orig_len, 2),
                clauses_retained=["đầu_văn_bản"],
                is_compressed=True,
            )

        keywords = self._extract_query_keywords(query)

        # Chấm điểm từng khoản
        scored_clauses: List[Tuple[int, str, str, float]] = []
        for idx, (label, c_text) in enumerate(clauses):
            s = self._score_clause(c_text, keywords, query)
            scored_clauses.append((idx, label, c_text, s))

        # Sắp xếp theo điểm giảm dần để chọn khoản trọng tâm nhất
        sorted_by_score = sorted(scored_clauses, key=lambda x: x[3], reverse=True)

        selected_indices: Set[int] = set()
        current_len = len(header)
        retained_labels: List[str] = ["Tiêu đề"]

        # Ưu tiên các khoản có điểm số > 0
        for orig_idx, label, c_text, score in sorted_by_score:
            if score <= 0.0 and len(selected_indices) >= 2:
                # Nếu đã có ít nhất 2 khoản tốt và khoản này điểm = 0 thì dừng
                break

            clause_len = len(c_text) + 20
            if current_len + clause_len <= budget or len(selected_indices) == 0:
                selected_indices.add(orig_idx)
                retained_labels.append(label)
                current_len += clause_len

        # Nếu không khoản nào có điểm > 0 (từ khóa không khớp trực tiếp), lấy 2 khoản đầu tiên
        if not selected_indices:
            for orig_idx in range(min(2, len(clauses))):
                selected_indices.add(orig_idx)
                retained_labels.append(clauses[orig_idx][0])

        # Ghép lại văn bản theo thứ tự thời gian gốc
        assembled_parts: List[str] = [header]
        sorted_selected = sorted(list(selected_indices))

        last_idx = -1
        for orig_idx in sorted_selected:
            if last_idx != -1 and orig_idx > last_idx + 1:
                assembled_parts.append("... [lược bớt các khoản không liên quan trực tiếp] ...")
            assembled_parts.append(clauses[orig_idx][1])
            last_idx = orig_idx

        if sorted_selected and sorted_selected[-1] < len(clauses) - 1:
            assembled_parts.append("... [lược bớt các khoản tiếp theo] ...")

        compressed_text = "\n\n".join(assembled_parts)
        comp_len = len(compressed_text)
        is_comp = (len(selected_indices) < len(clauses)) or (comp_len < orig_len)

        return CompressedPassage(
            original_text=article_text,
            compressed_text=compressed_text,
            original_length=orig_len,
            compressed_length=comp_len,
            compression_ratio=round(comp_len / orig_len, 2),
            clauses_retained=retained_labels,
            is_compressed=is_comp,
        )

    def compress_documents(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        max_chars_per_doc: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Compress a list of retrieved document dictionaries in-place with provenance."""
        compressed_docs: List[Dict[str, Any]] = []
        for d in docs:
            raw_text = d.get("text", "")
            cp = self.compress_article(
                query=query,
                article_text=raw_text,
                max_chars=max_chars_per_doc,
            )
            item = dict(d)
            item["text"] = cp.compressed_text
            item["raw_text"] = raw_text
            item["is_compressed"] = cp.is_compressed
            item["compression_ratio"] = cp.compression_ratio
            item["clauses_retained"] = cp.clauses_retained
            compressed_docs.append(item)
        return compressed_docs
