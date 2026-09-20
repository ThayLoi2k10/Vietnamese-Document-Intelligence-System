# src/rag/citation/validation/citation_validator.py
"""Bộ xác thực trích dẫn (CitationValidator) cho Phase 9 RAG.

Tuân thủ Mục 14: Citation Validation:
    - 14.1 Responsibility: validate(answer, evidence_registry) -> CitationValidationResult
    - 14.2 Kiểm tra cú pháp, sự tồn tại, đồng nhất danh tính và trích dẫn văn bản
    - 14.3 Tách bạch cú pháp với ngữ nghĩa (Syntactic vs Semantic)
    - 14.5 Xử lý trích dẫn không hợp lệ theo cấu hình (không tự động vá/thay thế)
"""

import re
from typing import Any, Dict, List, Optional, Union
import unicodedata

from src.rag.citation.provenance.evidence_registry import EvidenceRegistry
from src.rag.citation.validation.models import (
    CitationValidationResult,
    QuoteMatchingPolicy,
)
from src.rag.context.domain.models import ContextItem
from src.rag.errors.exceptions import InvalidCitationError
from src.rag.output.schemas.response_schema import Citation, LegalAnswer


class CitationValidator:
    """Xác thực tính toàn vẹn cú pháp và nguồn gốc của các trích dẫn pháp lý."""

    def __init__(
        self,
        quote_policy: QuoteMatchingPolicy = QuoteMatchingPolicy.SUBSTRING,
        raise_on_invalid: bool = False,
        component_version: str = "citation-val-v1.0",
    ) -> None:
        self.quote_policy = quote_policy
        self.raise_on_invalid = raise_on_invalid
        self.component_version = component_version

    def _normalize_string_for_fuzzy(self, text: str) -> str:
        """Chuẩn hóa văn bản cho so khớp quote có kiểm soát."""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text.strip().lower())
        text = re.sub(r"[.;:!?]+$", "", text)
        return text.strip()

    def _verify_quote(
        self, quote: Optional[str], source_text: str, policy: QuoteMatchingPolicy
    ) -> bool:
        """Kiểm tra xem đoạn trích có thực sự xuất hiện trong văn bản nguồn không."""
        if policy == QuoteMatchingPolicy.DISABLED or not quote:
            return True

        if policy == QuoteMatchingPolicy.EXACT:
            return quote in source_text

        if policy == QuoteMatchingPolicy.SUBSTRING:
            return quote.strip() in source_text

        if policy == QuoteMatchingPolicy.NORMALIZED_FUZZY:
            norm_quote = self._normalize_string_for_fuzzy(quote)
            norm_source = self._normalize_string_for_fuzzy(source_text)
            return norm_quote in norm_source

        return True

    def validate(
        self,
        answer: LegalAnswer,
        evidence_registry: Union[EvidenceRegistry, Dict[str, ContextItem]],
    ) -> CitationValidationResult:
        """Thực hiện xác thực toàn bộ danh sách trích dẫn trong câu trả lời."""
        # Chuẩn hóa registry
        if isinstance(evidence_registry, dict):
            reg = EvidenceRegistry()
            for it in evidence_registry.values():
                reg.register(it)
        else:
            reg = evidence_registry

        valid_citations: List[Citation] = []
        invalid_citations: List[Dict[str, Any]] = []
        errors: List[str] = []
        quote_matches: Dict[int, bool] = {}

        for c in answer.citations:
            c_num = c.citation_number
            c_errs: List[str] = []

            # 1. Kiểm tra citation_number hợp lệ (1-indexed)
            if c_num < 1:
                c_errs.append(f"Số hiệu trích dẫn không hợp lệ: {c_num} (< 1).")

            # 2. Kiểm tra sự tồn tại trong evidence registry
            matched_item = reg.lookup_by_number(c_num)
            if matched_item is None:
                c_errs.append(
                    f"Trích dẫn số [{c_num}] không tồn tại trong danh mục bằng chứng của prompt."
                )
            else:
                # 3. Kiểm tra danh tính nguồn (law_id)
                if c.law_id and matched_item.law_id:
                    # Cho phép so sánh mềm hoặc chính xác law_id
                    norm_c_law = c.law_id.strip().lower()
                    norm_item_law = matched_item.law_id.strip().lower()
                    if (
                        norm_c_law not in norm_item_law
                        and norm_item_law not in norm_c_law
                    ):
                        c_errs.append(
                            f"Trích dẫn [{c_num}] ghi nhận law_id='{c.law_id}' nhưng bằng chứng gốc có law_id='{matched_item.law_id}'."
                        )

                # 4. Kiểm tra quote matching nếu có
                if c.supporting_evidence_or_quote:
                    q_ok = self._verify_quote(
                        c.supporting_evidence_or_quote,
                        matched_item.text,
                        self.quote_policy,
                    )
                    quote_matches[c_num] = q_ok
                    if not q_ok:
                        c_errs.append(
                            f"Đoạn trích dẫn trong citation [{c_num}] không khớp với văn bản nguồn theo chính sách {self.quote_policy.value}."
                        )

            if c_errs:
                invalid_citations.append(
                    {
                        "citation": c.to_dict(),
                        "errors": c_errs,
                    }
                )
                errors.extend(c_errs)
            else:
                valid_citations.append(c)

        is_valid = len(invalid_citations) == 0

        if not is_valid and self.raise_on_invalid:
            raise InvalidCitationError(
                f"Phát hiện {len(invalid_citations)} trích dẫn không hợp lệ trong câu trả lời.",
                details={
                    "errors": errors,
                    "invalid_citations": invalid_citations,
                },
            )

        return CitationValidationResult(
            is_valid=is_valid,
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            errors=errors,
            quote_matches=quote_matches,
        )
