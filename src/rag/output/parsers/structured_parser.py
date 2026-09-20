# src/rag/output/parsers/structured_parser.py
"""Bộ phân tích cú pháp cấu trúc đầu ra (OutputParser) cho Phase 9 RAG.

Tuân thủ Mục 13: Structured Output:
    - 13.1 Public schema: JSON với answer, citations, confidence_level, abstention_reason
    - 13.4 Loại bỏ/không chấp nhận reasoning trace trong schema công khai
    - 13.5 Parsing: Phân biệt rõ valid vs malformed output
"""

import json
import re
from typing import Any, Dict

from src.rag.errors.exceptions import MalformedStructuredOutputError
from src.rag.output.schemas.response_schema import (
    Citation,
    ConfidenceLevel,
    LegalAnswer,
)


class OutputParser:
    """Parser kiểm tra và chuyển đổi văn bản sinh của LLM thành đối tượng LegalAnswer."""

    @staticmethod
    def _strip_markdown_code_fence(text: str) -> str:
        """Loại bỏ bọc markdown ```json ... ``` nếu có."""
        cleaned = text.strip()
        # Tìm khối ```json ... ``` hoặc ``` ... ```
        pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
        match = re.match(pattern, cleaned, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return cleaned

    def parse(self, raw_text: str) -> LegalAnswer:
        """Phân tích chuỗi JSON thô thành đối tượng LegalAnswer.

        Raises:
            MalformedStructuredOutputError: Khi cú pháp JSON hỏng hoặc thiếu trường bắt buộc.
        """
        if not raw_text or not raw_text.strip():
            raise MalformedStructuredOutputError(
                "Đầu ra thô của LLM rỗng, không thể phân tích cú pháp.",
                details={"raw_text": raw_text},
            )

        json_str = self._strip_markdown_code_fence(raw_text)

        if not json_str or not json_str.strip():
            raise MalformedStructuredOutputError(
                "Đầu ra JSON rỗng sau khi xử lý markdown code fence.",
                details={"raw_text": raw_text, "json_str": json_str},
            )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise MalformedStructuredOutputError(
                f"Lỗi cú pháp JSON khi phân tích đầu ra của LLM: {e}",
                details={"raw_text": raw_text, "json_str": json_str, "error": str(e)},
            )

        if not isinstance(data, dict):
            raise MalformedStructuredOutputError(
                f"Đầu ra JSON phải là một từ điển (dict/object), nhận được: {type(data).__name__}",
                details={"data": data, "raw_text": raw_text},
            )

        # Kiểm tra các trường bắt buộc
        for req_key in ["answer", "citations", "confidence_level"]:
            if req_key not in data:
                raise MalformedStructuredOutputError(
                    f"Đầu ra JSON thiếu trường bắt buộc '{req_key}'.",
                    details={"data": data, "raw_text": raw_text},
                )

        # Kiểm tra và parse enum confidence_level
        conf_str = str(data["confidence_level"]).strip().upper()
        try:
            confidence_level = ConfidenceLevel(conf_str)
        except ValueError:
            raise MalformedStructuredOutputError(
                f"Trường 'confidence_level' có giá trị không hợp lệ: '{conf_str}'. "
                f"Giá trị hợp lệ: {[c.value for c in ConfidenceLevel]}",
                details={"data": data, "raw_text": raw_text},
            )

        # Parse danh sách citations
        raw_citations = data.get("citations", [])
        if not isinstance(raw_citations, list):
            raise MalformedStructuredOutputError(
                f"Trường 'citations' phải là danh sách, nhận được: {type(raw_citations).__name__}",
                details={"data": data, "raw_text": raw_text},
            )

        parsed_citations = []
        for c in raw_citations:
            if not isinstance(c, dict):
                raise MalformedStructuredOutputError(
                    f"Mỗi phần tử trong 'citations' phải là dict, nhận được: {type(c).__name__}",
                    details={"citation": c, "raw_text": raw_text},
                )
            try:
                citation_obj = Citation.from_dict(c)
            except MalformedStructuredOutputError as c_err:
                c_err.details["raw_text"] = raw_text
                raise c_err
            parsed_citations.append(citation_obj)

        answer_text = str(data.get("answer", ""))
        abstention_reason = data.get("abstention_reason")
        if abstention_reason is not None:
            abstention_reason = str(abstention_reason)

        legal_answer = LegalAnswer(
            answer=answer_text,
            citations=parsed_citations,
            confidence_level=confidence_level,
            abstention_reason=abstention_reason,
        )
        try:
            legal_answer.validate()
        except MalformedStructuredOutputError as v_err:
            v_err.details["raw_text"] = raw_text
            raise v_err
        return legal_answer
