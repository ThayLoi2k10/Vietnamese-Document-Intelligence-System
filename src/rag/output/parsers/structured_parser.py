"""Robust parser converting raw LLM text into validated LegalAnswer objects."""

import ast
import json
import re
from typing import Any, Dict, List, Optional

from src.rag.errors.exceptions import MalformedStructuredOutputError
from src.rag.output.schemas.response_schema import (
    Citation,
    ConfidenceLevel,
    LegalAnswer,
)


class OutputParser:
    """Parses, sanitizes, and validates LLM generation outputs into LegalAnswer instances."""

    @staticmethod
    def _strip_markdown_code_fence(text: str) -> str:
        cleaned = text.strip()
        pattern = r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```$"
        match = re.match(pattern, cleaned, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)
            return cleaned.strip()
        return cleaned

    @staticmethod
    def _clean_trailing_commas(text: str) -> str:
        result = []
        in_string = False
        escape = False
        i = 0
        n = len(text)
        while i < n:
            char = text[i]
            if in_string:
                result.append(char)
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                    result.append(char)
                elif char == ',':
                    j = i + 1
                    while j < n and text[j].isspace():
                        j += 1
                    if j < n and text[j] in ('}', ']'):
                        pass
                    else:
                        result.append(char)
                else:
                    result.append(char)
            i += 1
        return "".join(result)

    @staticmethod
    def _clean_reasoning_tags(text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _extract_balanced_json_objects(text: str) -> List[str]:
        results = []
        i = 0
        n = len(text)
        while i < n:
            if text[i] == '{':
                start = i
                depth = 0
                in_string = False
                escape = False
                j = i
                while j < n:
                    char = text[j]
                    if in_string:
                        if escape:
                            escape = False
                        elif char == '\\':
                            escape = True
                        elif char == '"':
                            in_string = False
                    else:
                        if char == '"':
                            in_string = True
                        elif char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                results.append(text[start : j + 1])
                                i = j
                                break
                    j += 1
            i += 1
        return results

    def _extract_candidates(self, raw_text: str) -> List[str]:
        candidates = []

        def _add(cand: Optional[str]) -> None:
            if not cand:
                return
            cand_s = cand.strip()
            if cand_s and cand_s not in candidates:
                candidates.append(cand_s)

        cleaned_thinking = self._clean_reasoning_tags(raw_text)
        if cleaned_thinking and cleaned_thinking != raw_text.strip():
            _add(self._strip_markdown_code_fence(cleaned_thinking))
            _add(cleaned_thinking)

        _add(self._strip_markdown_code_fence(raw_text))

        fence_pattern = r"```[a-zA-Z0-9_-]*\s*\n?(.*?)(?:```|$)"
        fenced_blocks = re.findall(fence_pattern, raw_text, re.DOTALL | re.IGNORECASE)
        for fb in fenced_blocks:
            _add(fb)

        for bo in self._extract_balanced_json_objects(raw_text):
            _add(bo)

        first_b = raw_text.find("{")
        last_b = raw_text.rfind("}")
        if first_b != -1 and last_b != -1 and last_b > first_b:
            _add(raw_text[first_b : last_b + 1])

        _add(raw_text.strip())
        return candidates

    @staticmethod
    def _unwrap_and_normalize_data(data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return data

        if "answer" not in data:
            for val in data.values():
                if isinstance(val, dict) and "answer" in val:
                    data = dict(val)
                    break

        normalized = dict(data)

        if "answer" not in normalized:
            for k in ["content", "response", "answer_text", "text", "cau_tra_loi"]:
                if k in normalized and isinstance(normalized[k], (str, int, float, bool)):
                    normalized["answer"] = str(normalized[k])
                    break

        if "citations" not in normalized:
            for k in ["references", "sources", "cites", "trich_dan"]:
                if k in normalized:
                    normalized["citations"] = normalized[k]
                    break

        if "confidence_level" not in normalized:
            for k in ["confidence", "confidence_status", "muc_do_tin_cay"]:
                if k in normalized:
                    normalized["confidence_level"] = normalized[k]
                    break

        if "abstention_reason" not in normalized:
            for k in ["reason", "explanation", "ly_do"]:
                if k in normalized:
                    normalized["abstention_reason"] = normalized[k]
                    break

        return normalized

    def _build_legal_answer(self, data: Dict[str, Any], raw_text: str) -> LegalAnswer:
        data = self._unwrap_and_normalize_data(data)

        for req_key in ["answer", "citations", "confidence_level"]:
            if req_key not in data:
                raise MalformedStructuredOutputError(
                    f"JSON output missing required field '{req_key}'.",
                    details={"data": data, "raw_text": raw_text},
                )

        conf_val = data["confidence_level"]
        conf_str = str(conf_val).strip().upper() if conf_val is not None else "FULL"
        try:
            confidence_level = ConfidenceLevel(conf_str)
        except ValueError:
            if conf_str in ("FULL", "HIGH", "VERY_HIGH", "CONFIDENT", "CHẮC CHẮN", "ĐẦY ĐỦ", "TRUE", "1"):
                confidence_level = ConfidenceLevel.FULL
            elif conf_str in ("PARTIAL", "MEDIUM", "MODERATE", "MỘT PHẦN", "TƯƠNG ĐỐI", "2"):
                confidence_level = ConfidenceLevel.PARTIAL
            elif conf_str in ("INSUFFICIENT", "LOW", "NONE", "KHÔNG ĐỦ", "TỪ CHỐI", "FALSE", "0"):
                confidence_level = ConfidenceLevel.INSUFFICIENT
            else:
                raise MalformedStructuredOutputError(
                    f"Field 'confidence_level' has invalid value: '{conf_str}'. "
                    f"Valid choices: {[c.value for c in ConfidenceLevel]}",
                    details={"data": data, "raw_text": raw_text},
                )

        raw_citations = data.get("citations")
        if raw_citations is None or isinstance(raw_citations, bool):
            raw_citations = []
        elif isinstance(raw_citations, str):
            trimmed = raw_citations.strip()
            if not trimmed or trimmed.lower() in ("none", "null", "[]", "không có", "không", "n/a"):
                raw_citations = []
            else:
                raw_citations = [trimmed]
        elif isinstance(raw_citations, dict):
            raw_citations = [raw_citations]
        elif isinstance(raw_citations, (int, float)):
            raw_citations = [str(raw_citations)]
        elif not isinstance(raw_citations, list):
            raw_citations = []

        flattened_citations = []
        for item in raw_citations:
            if isinstance(item, list):
                flattened_citations.extend(item)
            else:
                flattened_citations.append(item)

        parsed_citations = []
        for idx, c in enumerate(flattened_citations, start=1):
            if c is None or isinstance(c, bool):
                continue

            if isinstance(c, str) or (isinstance(c, (int, float)) and not isinstance(c, bool)):
                c_str = str(c).strip()
                if not c_str or c_str.lower() in ("none", "null", "n/a", "[]"):
                    continue

                art_match = re.search(r"(Điều\s+\d+[a-zA-Z]*)", c_str, re.IGNORECASE)
                art_ident = art_match.group(1) if art_match else None

                parsed_citations.append(
                    Citation(
                        citation_number=idx,
                        law_id=c_str,
                        article_identifier=art_ident,
                        supporting_evidence_or_quote=c_str,
                    )
                )
                continue

            if isinstance(c, dict):
                c_dict = dict(c)
                if "citation_number" not in c_dict or not c_dict.get("citation_number"):
                    c_dict["citation_number"] = idx
                if "law_id" not in c_dict or not c_dict.get("law_id"):
                    c_dict["law_id"] = (
                        c_dict.get("article_identifier")
                        or c_dict.get("supporting_evidence_or_quote")
                        or c_dict.get("quote")
                        or c_dict.get("title")
                        or c_dict.get("document_id")
                        or "UNKNOWN"
                    )

                try:
                    parsed_citations.append(Citation.from_dict(c_dict))
                except Exception:
                    continue
                continue

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

    def _try_regex_reconstruction(self, text: str, raw_text: str) -> Optional[LegalAnswer]:
        if (
            "{" not in text
            or "}" not in text
            or "confidence_level" not in text
            or "citations" not in text
        ):
            return None
        ans_match = re.search(
            r'["\']answer["\']\s*:\s*["\'](.*?)(?:["\']\s*,\s*["\'](?:citations|confidence_level))',
            text,
            re.DOTALL,
        )
        if not ans_match:
            return None
        ans_text = ans_match.group(1).strip()
        conf_match = re.search(
            r'["\']confidence_level["\']\s*:\s*["\']?(FULL|PARTIAL|INSUFFICIENT)["\']?',
            text,
            re.IGNORECASE,
        )
        if not conf_match:
            return None
        conf_str = conf_match.group(1).upper()
        abs_match = re.search(
            r'["\']abstention_reason["\']\s*:\s*(?:["\'](.*?)["\']|null)', text, re.DOTALL
        )
        abs_reason = abs_match.group(1) if abs_match and abs_match.group(1) else None

        cit_match = re.search(r'["\']citations["\']\s*:\s*(\[.*?\])', text, re.DOTALL)
        raw_cits = []
        if cit_match:
            try:
                raw_cits = json.loads(cit_match.group(1), strict=False)
            except Exception:
                raw_cits = []

        try:
            return self._build_legal_answer(
                {
                    "answer": ans_text,
                    "citations": raw_cits,
                    "confidence_level": conf_str,
                    "abstention_reason": abs_reason,
                },
                raw_text,
            )
        except Exception:
            return None

    def parse(self, raw_text: str) -> LegalAnswer:
        if not raw_text or not raw_text.strip():
            raise MalformedStructuredOutputError(
                "Raw LLM output is empty.",
                details={"raw_text": raw_text},
            )

        json_str_legacy = self._strip_markdown_code_fence(raw_text)
        if not json_str_legacy or not json_str_legacy.strip():
            raise MalformedStructuredOutputError(
                "JSON content is empty after stripping markdown fences.",
                details={"raw_text": raw_text, "json_str": json_str_legacy},
            )

        candidates = self._extract_candidates(raw_text)
        last_error = None
        last_candidate = json_str_legacy
        non_dict_data = None
        dict_candidates = []
        last_validation_error = None

        for cand in candidates:
            for text_to_try in [cand, self._clean_trailing_commas(cand)]:
                last_candidate = text_to_try
                try:
                    data = json.loads(text_to_try, strict=False)
                except json.JSONDecodeError as err:
                    last_error = err
                    try:
                        data = ast.literal_eval(text_to_try)
                    except Exception:
                        continue

                if isinstance(data, dict):
                    normalized_data = self._unwrap_and_normalize_data(data)
                    if "answer" in normalized_data and "confidence_level" in normalized_data and "citations" in normalized_data:
                        try:
                            return self._build_legal_answer(normalized_data, raw_text)
                        except MalformedStructuredOutputError as val_err:
                            last_validation_error = val_err
                    else:
                        dict_candidates.append(normalized_data)
                    break
                else:
                    if non_dict_data is None:
                        non_dict_data = data
                    break

        for cand in candidates:
            reconstructed = self._try_regex_reconstruction(cand, raw_text)
            if reconstructed is not None:
                return reconstructed

        if last_validation_error is not None:
            raise last_validation_error

        if dict_candidates:
            return self._build_legal_answer(dict_candidates[0], raw_text)

        if non_dict_data is not None:
            raise MalformedStructuredOutputError(
                f"JSON output must be a dictionary/object, got: {type(non_dict_data).__name__}",
                details={"data": non_dict_data, "raw_text": raw_text},
            )

        if "{" not in raw_text:
            clean_plain_text = self._strip_markdown_code_fence(raw_text).strip()
            if not (clean_plain_text.startswith("[") and clean_plain_text.endswith("]")) and clean_plain_text:
                return LegalAnswer(
                    answer=clean_plain_text,
                    citations=[],
                    confidence_level=ConfidenceLevel.FULL,
                    abstention_reason=None,
                )

        raise MalformedStructuredOutputError(
            f"JSON decode error parsing LLM response: {last_error}",
            details={"raw_text": raw_text, "json_str": last_candidate, "error": str(last_error)},
        )
