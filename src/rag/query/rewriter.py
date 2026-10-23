"""LLM-based Vietnamese legal query rewriter and terminology expander."""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set

from src.rag.query.models import RewritingResult

logger = logging.getLogger(__name__)

VI_STOPWORDS: Set[str] = {
    "và", "của", "có", "được", "cho", "là", "những", "các", "trong", "khi",
    "với", "về", "như", "thế", "nào", "gì", "ở", "tại", "theo", "thì",
    "ra", "vào", "lại", "đến", "để", "bị", "phải", "không", "hay", "hoặc",
    "ai", "đâu", "sao", "bao", "nhiêu", "mấy",
}


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class LLMQueryRewriter:
    """Legal query rewriter utilizing an LLM for terminology expansion."""

    DEFAULT_SYSTEM_PROMPT = """Bạn là chuyên gia về Hệ thống Pháp luật Việt Nam và Kỹ thuật Truy xuất Thông tin (Information Retrieval).
Nhiệm vụ của bạn là phân tích câu hỏi pháp lý của người dùng, xác định ý định pháp lý cốt lõi và bổ sung các THUẬT NGỮ PHÁP LÝ CHUẨN XÁC theo văn bản quy phạm pháp luật Việt Nam.

NGUYÊN TẮC BẮT BUỘC (CONSERVATIVE EXPANSION):
1. GIỮ NGUYÊN Ý ĐỊNH VÀ TỪ KHÓA GỐC: Tuyệt đối không thay đổi bản chất câu hỏi, không làm mất các thực thể hoặc sự kiện trong câu gốc.
2. MỞ RỘNG CÓ KIỂM SOÁT (TERMINOLOGY EXPANSION): Chỉ bổ sung các thuật ngữ pháp lý chính thức tương đương hoặc bao hàm trong luật (ví dụ: "đuổi việc" -> "đơn phương chấm dứt hợp đồng lao động", "sổ đỏ" -> "giấy chứng nhận quyền sử dụng đất").
3. KHÔNG TỰ BỊA ĐẶT CĂN CỨ HAY TÌNH TIẾT: Không tự ý thêm số hiệu điều luật, không thêm tình tiết giả định mà câu gốc không nêu.
4. ĐỊNH DẠNG ĐẦU RA BẮT BUỘC: Trả về duy nhất một khối JSON hợp lệ theo cấu trúc sau:
```json
{
  "intent": "Tóm tắt ngắn gọn ý định tra cứu pháp lý",
  "statutory_terms": ["thuật ngữ 1", "thuật ngữ 2"],
  "expanded_query": "Câu truy vấn hoàn chỉnh kết hợp từ khóa gốc và các thuật ngữ pháp lý mở rộng",
  "confidence": "HIGH | MEDIUM | LOW"
}
```"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        prompt_config_path: Optional[str] = None,
        inference_fn: Optional[Callable[[str], str]] = None,
        system_prompt: Optional[str] = None,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.inference_fn = inference_fn
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.prompt_id = "query_rewriter_v1"

        if prompt_config_path:
            self._load_config(prompt_config_path)
        else:
            self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
            self.few_shot_examples = few_shot_examples or self._default_few_shot()

    def _load_config(self, config_path: str) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.prompt_id = cfg.get("prompt_id", "query_rewriter_v1")
        self.system_prompt = cfg.get("system_prompt", self.DEFAULT_SYSTEM_PROMPT)
        self.few_shot_examples = cfg.get("few_shot_examples", self._default_few_shot())

    def _default_few_shot(self) -> List[Dict[str, Any]]:
        return [
            {
                "input": "Công ty tự ý đuổi việc người lao động thì phải bồi thường những gì?",
                "output": {
                    "intent": "Bồi thường khi người sử dụng lao động đơn phương chấm dứt hợp đồng trái pháp luật",
                    "statutory_terms": [
                        "đơn phương chấm dứt hợp đồng lao động trái pháp luật",
                        "nghĩa vụ của người sử dụng lao động",
                        "tiền lương",
                        "trợ cấp thôi việc",
                    ],
                    "expanded_query": "Công ty tự ý đuổi việc người lao động thì phải bồi thường những gì đơn phương chấm dứt hợp đồng lao động trái pháp luật tiền lương trợ cấp",
                    "confidence": "HIGH",
                },
            },
            {
                "input": "Thủ tục sang tên sổ đỏ đất nông nghiệp tại xã?",
                "output": {
                    "intent": "Thủ tục đăng ký biến động chuyển nhượng quyền sử dụng đất nông nghiệp",
                    "statutory_terms": [
                        "đăng ký biến động đất đai",
                        "giấy chứng nhận quyền sử dụng đất",
                        "chuyển nhượng quyền sử dụng đất nông nghiệp",
                    ],
                    "expanded_query": "Thủ tục sang tên sổ đỏ đất nông nghiệp tại xã đăng ký biến động giấy chứng nhận quyền sử dụng đất chuyển nhượng",
                    "confidence": "HIGH",
                },
            },
        ]

    def build_prompt(self, query: str) -> str:
        lines = [self.system_prompt, "\n--- VÍ DỤ MINH HỌA ---"]
        for ex in self.few_shot_examples:
            lines.append(f"Câu hỏi: {ex['input']}")
            lines.append("Kết quả JSON:")
            lines.append(json.dumps(ex["output"], ensure_ascii=False, indent=2))
            lines.append("")
        lines.append("--- THỰC THI TRÊN CÂU HỎI MỚI ---")
        lines.append(f"Câu hỏi: {query}")
        lines.append("Kết quả JSON:")
        return "\n".join(lines)

    def rewrite(self, query: str) -> RewritingResult:
        if not query or not query.strip():
            return RewritingResult(
                is_valid=False, error_message="Empty input query", expanded_query=query
            )

        if self.inference_fn is None:
            return RewritingResult(
                intent="No LLM backend configured",
                statutory_terms=[],
                expanded_query=query,
                confidence="LOW",
                is_valid=True,
            )

        prompt = self.build_prompt(query)
        try:
            raw_response = self.inference_fn(prompt)
            return self.parse_and_validate(query, raw_response)
        except Exception as exc:
            logger.warning("LLM inference error: %s", exc)
            return RewritingResult(
                is_valid=False,
                error_message=f"Inference exception: {str(exc)}",
                expanded_query=query,
            )

    def parse_and_validate(self, original_query: str, raw_llm_output: str) -> RewritingResult:
        if not raw_llm_output or not raw_llm_output.strip():
            return RewritingResult(
                is_valid=False,
                error_message="Raw LLM output is empty",
                expanded_query=original_query,
            )

        json_str = raw_llm_output.strip()
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_llm_output, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        else:
            first_brace = raw_llm_output.find("{")
            last_brace = raw_llm_output.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = raw_llm_output[first_brace : last_brace + 1].strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as err:
            return RewritingResult(
                is_valid=False,
                error_message=f"JSON parsing error: {err}",
                expanded_query=original_query,
            )

        intent = str(data.get("intent", "")).strip()
        statutory_terms = data.get("statutory_terms", [])
        if isinstance(statutory_terms, str):
            statutory_terms = [statutory_terms]
        elif not isinstance(statutory_terms, list):
            statutory_terms = []
        statutory_terms = [str(t).strip() for t in statutory_terms if str(t).strip()]

        expanded_query = str(data.get("expanded_query", "")).strip()
        confidence = str(data.get("confidence", "MEDIUM")).strip().upper()
        if confidence not in {"HIGH", "MEDIUM", "LOW"}:
            confidence = "MEDIUM"

        if not expanded_query or len(expanded_query) < 5:
            return RewritingResult(
                is_valid=False,
                error_message="Expanded query is empty or too short (<5 chars)",
                expanded_query=original_query,
            )

        orig_tokens = set(tokenize_words(original_query)) - VI_STOPWORDS
        expanded_tokens = set(tokenize_words(expanded_query))

        if orig_tokens:
            preserved_ratio = len(orig_tokens.intersection(expanded_tokens)) / len(orig_tokens)
            if preserved_ratio < 0.50:
                return RewritingResult(
                    intent=intent,
                    statutory_terms=statutory_terms,
                    expanded_query=original_query,
                    confidence=confidence,
                    is_valid=False,
                    error_message=f"Information loss: only {preserved_ratio*100:.1f}% key tokens preserved (<50%)",
                )

        return RewritingResult(
            intent=intent,
            statutory_terms=statutory_terms,
            expanded_query=expanded_query,
            confidence=confidence,
            is_valid=True,
            error_message=None,
        )
