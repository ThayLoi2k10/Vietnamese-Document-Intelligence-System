"""Mock LLM generator simulating varied responses and failure modes."""

import json
from typing import Any, Dict, Optional

from src.rag.errors.exceptions import LLMGenerationError
from src.rag.generation.domain.models import (
    FinishStatus,
    GenerationRequest,
    GenerationResult,
)
from src.rag.generation.interfaces.generator import LLMGenerator


class MockLLMGenerator(LLMGenerator):
    """Configurable mock generator returning predefined response scenarios."""

    MODE_VALID_ANSWER = "valid_answer"
    MODE_MALFORMED_JSON = "malformed_json"
    MODE_INVALID_CITATION = "invalid_citation"
    MODE_ABSTENTION = "abstention"
    MODE_PARTIAL_ANSWER = "partial_answer"
    MODE_FAILURE = "failure"

    def __init__(
        self,
        mode: str = MODE_VALID_ANSWER,
        custom_raw_response: Optional[str] = None,
        custom_json_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.mode = mode
        self.custom_raw_response = custom_raw_response
        self.custom_json_data = custom_json_data
        self.call_count = 0
        self.last_request: Optional[GenerationRequest] = None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.call_count += 1
        self.last_request = request

        if self.mode == self.MODE_FAILURE:
            raise LLMGenerationError(
                "MockLLMGenerator simulating connection or service failure.",
                component_identifier=request.model_identifier,
            )

        if self.custom_raw_response is not None:
            raw_text = self.custom_raw_response
        elif self.custom_json_data is not None:
            raw_text = json.dumps(self.custom_json_data, ensure_ascii=False)
        else:
            raw_text = self._build_preset_response(self.mode, request)

        return GenerationResult(
            model_identifier=request.model_identifier,
            raw_response=raw_text,
            generation_metadata={
                "mock_mode": self.mode,
                "temperature": request.generation_parameters.temperature,
                "thinking_mode": request.thinking_mode,
            },
            usage_metadata={
                "prompt_tokens": len(request.prompt.user_prompt.split()),
                "completion_tokens": len(raw_text.split()),
                "total_tokens": len(request.prompt.user_prompt.split())
                + len(raw_text.split()),
            },
            finish_status=FinishStatus.COMPLETED,
        )

    def _build_preset_response(
        self, mode: str, request: GenerationRequest
    ) -> str:
        evidence_items = list(request.prompt.evidence_registry.values())
        first_law_id = evidence_items[0].law_id if evidence_items else "LAW_DEFAULT"
        first_art = (
            evidence_items[0].article_identifier if evidence_items else "Điều 1"
        )

        if mode == self.MODE_VALID_ANSWER:
            data = {
                "answer": "Người tham gia giao thông phải chấp hành hiệu lệnh theo quy định.",
                "citations": [
                    {
                        "citation_number": 1,
                        "law_id": first_law_id,
                        "article_identifier": first_art,
                        "supporting_evidence_or_quote": "chấp hành hiệu lệnh",
                    }
                ],
                "confidence_level": "FULL",
                "abstention_reason": None,
            }
            return json.dumps(data, ensure_ascii=False)

        if mode == self.MODE_MALFORMED_JSON:
            return "Đây là câu trả lời không đúng định dạng JSON: { 'answer': lỗi cú pháp json ... "

        if mode == self.MODE_INVALID_CITATION:
            data = {
                "answer": "Căn cứ theo quy định tại luật không hề tồn tại trong tài liệu.",
                "citations": [
                    {
                        "citation_number": 999,
                        "law_id": "LUAT_MA_KHONG_CO_TRONG_CONTEXT",
                        "article_identifier": "Điều 999",
                        "supporting_evidence_or_quote": "nội dung ma",
                    }
                ],
                "confidence_level": "FULL",
                "abstention_reason": None,
            }
            return json.dumps(data, ensure_ascii=False)

        if mode == self.MODE_ABSTENTION:
            data = {
                "answer": "Tài liệu căn cứ pháp lý được cung cấp không đủ thông tin để trả lời câu hỏi này.",
                "citations": [],
                "confidence_level": "INSUFFICIENT",
                "abstention_reason": "Tài liệu căn cứ pháp lý không đề cập đến trường hợp được hỏi.",
            }
            return json.dumps(data, ensure_ascii=False)

        if mode == self.MODE_PARTIAL_ANSWER:
            data = {
                "answer": "Tài liệu chỉ quy định về nguyên tắc chung, chưa có hướng dẫn chi tiết về trường hợp này.",
                "citations": [
                    {
                        "citation_number": 1,
                        "law_id": first_law_id,
                        "article_identifier": first_art,
                        "supporting_evidence_or_quote": "nguyên tắc chung",
                    }
                ],
                "confidence_level": "PARTIAL",
                "abstention_reason": "Chưa có quy định chi tiết về thủ tục",
            }
            return json.dumps(data, ensure_ascii=False)

        return "{}"
