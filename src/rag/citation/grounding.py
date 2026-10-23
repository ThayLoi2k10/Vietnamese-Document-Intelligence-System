"""Grounding and citation verification engine cross-referencing generated legal claims with retrieved documents."""

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List, Optional, Set
import unicodedata


@dataclass
class GroundingResult:
    """Detailed verification outcome of generated legal text against retrieved context."""

    is_grounded: bool
    verdict: str  # 'VERIFIED', 'PARTIAL', 'HALLUCINATION_RISK', 'ABSTAINED', 'UNGROUNDED'
    grounding_score: float  # 0.0 to 1.0
    badge: str
    verified_laws: List[str] = field(default_factory=list)
    unverified_laws: List[str] = field(default_factory=list)
    verified_articles: List[str] = field(default_factory=list)
    unverified_articles: List[str] = field(default_factory=list)
    quote_verifications: List[Dict[str, Any]] = field(default_factory=list)
    cited_sources: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GroundingVerifier:
    """Verifies citations, statutory codes, and quotes in LLM responses against source evidence."""

    FORMAL_LAW_PATTERN = re.compile(
        r"\b\d{1,4}/\d{4}/(?:QH\d+|NĐ-CP|TT-[A-ZĐ]+|QĐ-[A-ZĐ]+|NQ-[A-ZĐ]+|PL-UBTVQH\d*)\b",
        re.IGNORECASE,
    )

    NAMED_LAW_PATTERNS = [
        "Bộ luật Lao động",
        "Bộ luật Dân sự",
        "Bộ luật Hình sự",
        "Bộ luật Tố tụng dân sự",
        "Bộ luật Tố tụng hình sự",
        "Luật Đất đai",
        "Luật Doanh nghiệp",
        "Luật Thương mại",
        "Luật Bảo hiểm xã hội",
        "Luật Hôn nhân và gia đình",
        "Luật An toàn, vệ sinh lao động",
        "Luật Giao thông đường bộ",
        "Luật Xây dựng",
        "Luật Nhà ở",
        "Luật Đầu tư",
        "Luật Cán bộ, công chức",
        "Luật Viên chức",
        "Luật Thuế thu nhập cá nhân",
        "Luật Quản lý thuế",
    ]

    MAJOR_LAW_MAPPING: Dict[str, List[str]] = {
        "bộ luật lao động": ["45/2019/qh14", "10/2012/qh13"],
        "bộ luật dân sự": ["91/2015/qh13", "33/2005/qh11"],
        "bộ luật hình sự": ["100/2015/qh13", "12/2017/qh14"],
        "bộ luật tố tụng dân sự": ["92/2015/qh13"],
        "bộ luật tố tụng hình sự": ["101/2015/qh13"],
        "luật doanh nghiệp": ["59/2020/qh14", "68/2014/qh13"],
        "luật đất đai": ["31/2024/qh15", "45/2013/qh13"],
        "luật thương mại": ["36/2005/qh11"],
        "luật bảo hiểm xã hội": ["41/2024/qh15", "58/2014/qh13"],
        "luật hôn nhân và gia đình": ["52/2014/qh13"],
        "luật an toàn, vệ sinh lao động": ["84/2015/qh13"],
        "luật giao thông đường bộ": ["23/2008/qh12", "36/2024/qh15"],
        "luật nhà ở": ["27/2023/qh15", "65/2014/qh13"],
        "luật kinh doanh bất động sản": ["29/2023/qh15", "66/2014/qh13"],
        "luật đầu tư": ["61/2020/qh14", "67/2014/qh13"],
        "luật cán bộ, công chức": ["22/2008/qh12", "52/2019/qh14"],
        "luật viên chức": ["58/2010/qh12"],
        "luật quản lý thuế": ["38/2019/qh14"],
        "luật thuế thu nhập cá nhân": ["04/2007/qh12"],
    }

    ARTICLE_PATTERN = re.compile(r"\bĐiều\s+(\d+[a-z]?)\b", re.IGNORECASE)
    DOC_REF_PATTERN = re.compile(
        r"\[(?:Tài liệu|TÀI LIỆU|Doc|Document)?\s*#?(\d+)\]", re.IGNORECASE
    )
    QUOTE_PATTERN = re.compile(r'["“]([^"”]{8,250})["”]')

    def __init__(self, quote_tolerance: float = 0.85) -> None:
        self.quote_tolerance = quote_tolerance

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize Unicode, lower case, and collapse whitespace for resilient matching."""
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text.strip().lower())
        return text

    def extract_formal_laws(self, text: str) -> List[str]:
        """Extract official Vietnamese legal document identifiers (e.g., 45/2019/QH14)."""
        matches = self.FORMAL_LAW_PATTERN.findall(text)
        seen: Set[str] = set()
        results: List[str] = []
        for m in matches:
            cleaned = m.strip().upper()
            if cleaned not in seen:
                seen.add(cleaned)
                results.append(cleaned)
        return results

    def extract_named_laws(self, text: str) -> List[str]:
        """Extract common statutory names mentioned in text."""
        results: List[str] = []
        for name in self.NAMED_LAW_PATTERNS:
            pattern = re.compile(
                rf"\b{re.escape(name)}(?:\s+năm\s+\d{{4}}|\s+\d{{4}})?\b",
                re.IGNORECASE,
            )
            found = pattern.findall(text)
            if found:
                results.append(found[0].strip())
        return results

    def extract_articles(self, text: str) -> List[str]:
        """Extract statutory article numbers (e.g., Điều 25, Điều 107)."""
        matches = self.ARTICLE_PATTERN.findall(text)
        seen: Set[str] = set()
        results: List[str] = []
        for num in matches:
            art_label = f"Điều {num}"
            if art_label.lower() not in seen:
                seen.add(art_label.lower())
                results.append(art_label)
        return results

    def extract_quotes(self, text: str) -> List[str]:
        """Extract explicit textual quotations enclosed in quotes."""
        matches = self.QUOTE_PATTERN.findall(text)
        return [q.strip() for q in matches if len(q.strip()) >= 8]

    def verify(
        self,
        response_text: str,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None,
    ) -> GroundingResult:
        """Verify statutory citations, articles, and quotes against retrieved documents."""
        docs = retrieved_docs or []
        if not response_text or not response_text.strip():
            return GroundingResult(
                is_grounded=True,
                verdict="ABSTAINED",
                grounding_score=1.0,
                badge="⚪ KHÔNG CÓ DỮ LIỆU",
                summary="Không có nội dung phản hồi cần kiểm chứng.",
            )

        # 1. Kiểm tra trường hợp từ chối hợp lệ (Abstention)
        is_abstention = (
            "THÔNG BÁO THIẾU CĂN CỨ" in response_text
            or "không có đủ căn cứ" in response_text.lower()
            or "không có đủ thông tin" in response_text.lower()
        )
        if is_abstention:
            return GroundingResult(
                is_grounded=True,
                verdict="ABSTAINED",
                grounding_score=1.0,
                badge="⚪ TỪ CHỐI DO THIẾU CĂN CỨ (Hợp lệ)",
                summary="Hệ thống đã từ chối trả lời do tài liệu truy xuất không đủ căn cứ pháp lý.",
            )

        # 2. Thu thập ngữ cảnh nguồn từ retrieved_docs
        source_texts: List[str] = []
        source_law_ids: Set[str] = set()
        source_articles: Set[str] = set()
        cited_sources: List[Dict[str, Any]] = []

        for d in docs:
            raw_text = d.get("text", "")
            norm_doc = self._normalize_text(raw_text)
            source_texts.append(norm_doc)

            law_id = str(d.get("law_id", "")).strip().upper()
            if law_id and law_id != "NONE" and law_id != "N/A":
                source_law_ids.add(law_id)

            aid = str(d.get("aid", "")).strip()
            if aid:
                source_articles.add(aid.lower())
                source_articles.add(f"điều {aid.lower()}")

            # Tìm số Điều trong chính đoạn trích text của doc
            art_matches = self.ARTICLE_PATTERN.findall(raw_text)
            for num in art_matches:
                source_articles.add(f"điều {num.lower()}")
                source_articles.add(num.lower())

            cited_sources.append(
                {
                    "rank": d.get("rank"),
                    "law_id": d.get("law_id"),
                    "article": d.get("aid"),
                    "score": d.get("score"),
                }
            )

        combined_sources_text = " ".join(source_texts)

        # 3. Trích xuất căn cứ từ câu trả lời của LLM
        formal_laws = self.extract_formal_laws(response_text)
        named_laws = self.extract_named_laws(response_text)
        articles = self.extract_articles(response_text)
        quotes = self.extract_quotes(response_text)

        # 4. Đối soát Văn bản Luật
        verified_laws: List[str] = []
        unverified_laws: List[str] = []

        all_laws = formal_laws + named_laws
        seen_laws: Set[str] = set()
        for law in all_laws:
            norm_law = self._normalize_text(law)
            if norm_law in seen_laws:
                continue
            seen_laws.add(norm_law)

            # Khớp qua law_id hoặc xuất hiện trong văn bản nguồn
            law_matched = False
            for s_lid in source_law_ids:
                if norm_law in s_lid.lower() or s_lid.lower() in norm_law:
                    law_matched = True
                    break

            if not law_matched:
                if norm_law in combined_sources_text:
                    law_matched = True

            # Khớp qua MAJOR_LAW_MAPPING (ví dụ: "Bộ luật Lao động" ứng với "45/2019/QH14")
            if not law_matched:
                for key, qh_ids in self.MAJOR_LAW_MAPPING.items():
                    if key in norm_law or norm_law in key:
                        for qh in qh_ids:
                            if any(qh in s.lower() for s in source_law_ids):
                                law_matched = True
                                break
                    if law_matched:
                        break

            if law_matched:
                verified_laws.append(law)
            else:
                unverified_laws.append(law)

        # 5. Đối soát Điều luật
        verified_articles: List[str] = []
        unverified_articles: List[str] = []

        for art in articles:
            norm_art = art.lower().strip()
            # Trích xuất số điều
            num_match = re.search(r"\d+[a-z]?", norm_art)
            num_str = num_match.group(0) if num_match else norm_art

            art_matched = False
            if norm_art in source_articles or num_str in source_articles:
                art_matched = True
            elif norm_art in combined_sources_text:
                art_matched = True

            if art_matched:
                verified_articles.append(art)
            else:
                unverified_articles.append(art)

        # 6. Đối soát Trích dẫn nguyên văn (Quotes)
        quote_verifications: List[Dict[str, Any]] = []
        for q in quotes:
            norm_q = self._normalize_text(q)
            q_matched = False
            matched_rank = None

            for idx, doc_text in enumerate(source_texts, start=1):
                if norm_q in doc_text:
                    q_matched = True
                    matched_rank = idx
                    break

            quote_verifications.append(
                {
                    "quote": q,
                    "matched": q_matched,
                    "matched_doc_rank": matched_rank,
                }
            )

        # 7. Tính điểm Grounding Score & Phán quyết
        total_claims = (
            len(verified_laws)
            + len(unverified_laws)
            + len(verified_articles)
            + len(unverified_articles)
        )
        if total_claims == 0:
            if quotes:
                matched_quotes = sum(1 for qv in quote_verifications if qv["matched"])
                grounding_score = round(matched_quotes / len(quotes), 2)
            else:
                # Không trích dẫn điều luật cụ thể nào nhưng có câu trả lời
                grounding_score = 0.7
        else:
            verified_claims = len(verified_laws) + len(verified_articles)
            grounding_score = round(verified_claims / total_claims, 2)

        has_hallucination = bool(unverified_laws or unverified_articles)
        all_quotes_matched = all(qv["matched"] for qv in quote_verifications) if quotes else True

        if has_hallucination:
            verdict = "HALLUCINATION_RISK"
            is_grounded = False
            badge = f"🔴 CẢNH BÁO ẢO GIÁC ({int(grounding_score * 100)}% khớp)"
            err_items = unverified_laws + unverified_articles
            summary = (
                f"Phát hiện căn cứ không có trong tài liệu trích xuất: {', '.join(err_items)}. "
                "Có nguy cơ mô hình tự suy diễn điều luật (Hallucination)."
            )
        elif grounding_score >= 0.85 and all_quotes_matched:
            verdict = "VERIFIED"
            is_grounded = True
            badge = f"🟢 ĐÃ KIỂM CHỨNG ({int(grounding_score * 100)}%)"
            summary = (
                f"Toàn bộ căn cứ pháp lý ({len(verified_laws)} văn bản, {len(verified_articles)} điều) "
                "và các trích dẫn nguyên văn đều khớp chính xác với dữ liệu văn bản gốc."
            )
        elif grounding_score >= 0.5:
            verdict = "PARTIAL"
            is_grounded = True
            badge = f"🟡 CẦN LƯU Ý ({int(grounding_score * 100)}%)"
            summary = "Một số trích dẫn hoặc câu chữ chưa được đối soát 100% với tài liệu nguồn."
        else:
            verdict = "UNGROUNDED"
            is_grounded = False
            badge = f"🔴 THIẾU CĂN CỨ ({int(grounding_score * 100)}%)"
            summary = "Nội dung trả lời chưa có đủ bằng chứng xác thực từ tài liệu trích xuất."

        return GroundingResult(
            is_grounded=is_grounded,
            verdict=verdict,
            grounding_score=grounding_score,
            badge=badge,
            verified_laws=verified_laws,
            unverified_laws=unverified_laws,
            verified_articles=verified_articles,
            unverified_articles=unverified_articles,
            quote_verifications=quote_verifications,
            cited_sources=cited_sources,
            summary=summary,
        )
