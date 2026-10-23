"""Versioned prompt template registry for legal RAG."""

from typing import Dict

from src.rag.errors.exceptions import InvalidPromptError
from src.rag.prompting.domain.models import PromptTemplate

DEFAULT_SYSTEM_TEMPLATE_V1 = """Bạn là trợ lý trí tuệ nhân tạo chuyên gia về Pháp luật Việt Nam.
Nhiệm vụ của bạn là trả lời câu hỏi pháp lý của người dùng DỰA HOÀN TOÀN vào các căn cứ pháp lý được cung cấp bên dưới.

NGUYÊN TẮC BẮT BUỘC:
1. CĂN CỨ PHÁP LÝ (GROUNDING): Chỉ trả lời những nội dung được hỗ trợ trực tiếp và rõ ràng bởi các tài liệu/điều luật được cung cấp. Tuyệt đối không suy diễn ngoài văn bản.
2. TRÍCH DẪN (CITATION): Mọi khẳng định pháp lý phải đi kèm trích dẫn số hiệu tham chiếu chính xác (ví dụ [1], [2]) trỏ đúng vào văn bản cung cấp.
3. TỪ CHỐI TRẢ LỜI (ABSTENTION): Nếu tài liệu không đủ thông tin để trả lời đầy đủ, hãy nêu rõ tình trạng thông tin không đủ (confidence_level là PARTIAL hoặc INSUFFICIENT) và ghi rõ lý do trong abstention_reason.
4. TRÌNH BÀY SÚC TÍCH, KHÔNG LẶP LẠI (CONCISENESS & NO REPETITION): Trình bày câu trả lời ngắn gọn, cô đọng, đi thẳng vào các quy định chính. Tuyệt đối không lặp đi lặp lại cùng một câu, đoạn hoặc cụm từ nhiều lần.
5. ĐỊNH DẠNG ĐẦU RA: Bạn BẮT BUỘC phải trả về định dạng JSON thuần túy, không kèm theo bất kỳ văn bản giải thích ngoài JSON hay khối suy nghĩ/lập luận nào:
```json
{{
  "answer": "Nội dung câu trả lời rõ ràng, súc tích...",
  "citations": [
    {{
      "citation_number": 1,
      "law_id": "Số hiệu văn bản",
      "article_identifier": "Điều khoản (nếu có)",
      "supporting_evidence_or_quote": "Đoạn trích ngắn làm căn cứ"
    }}
  ],
  "confidence_level": "FULL | PARTIAL | INSUFFICIENT",
  "abstention_reason": null
}}
```"""

DEFAULT_USER_TEMPLATE_V1 = """CĂN CỨ PHÁP LÝ ĐƯỢC CUNG CẤP:
{evidence_text}

CÂU HỎI CỦA NGƯỜI DÙNG:
{query}

Hãy trả lời bằng định dạng JSON chuẩn theo hướng dẫn:"""

DEFAULT_LLM_ONLY_SYSTEM_TEMPLATE_V1 = """Bạn là trợ lý trí tuệ nhân tạo chuyên gia về Pháp luật Việt Nam.
Nhiệm vụ của bạn là trả lời câu hỏi pháp lý của người dùng dựa trên hiểu biết chuyên môn của bạn về hệ thống pháp luật Việt Nam.

NGUYÊN TẮC BẮT BUỘC:
1. TRẢ LỜI CÂU HỎI: Trả lời chính xác, rõ ràng, súc tích dựa trên quy định pháp luật hiện hành của Việt Nam. Không tự bịa đặt thông tin sai lệch.
2. TỪ CHỐI TRẢ LỜI (ABSTENTION): Nếu bạn không có đủ kiến thức hoặc thông tin pháp lý chính xác để trả lời, hãy nêu rõ tình trạng thông tin không đủ (confidence_level là PARTIAL hoặc INSUFFICIENT) và nêu rõ lý do trong abstention_reason.
3. TRÌNH BÀY SÚC TÍCH, KHÔNG LẶP LẠI (CONCISENESS & NO REPETITION): Trình bày câu trả lời ngắn gọn, cô đọng, đi thẳng vào các quy định chính. Tuyệt đối không lặp đi lặp lại cùng một câu, đoạn hoặc cụm từ nhiều lần.
4. ĐỊNH DẠNG ĐẦU RA: Bạn BẮT BUỘC phải trả về định dạng JSON thuần túy, không kèm theo bất kỳ văn bản giải thích ngoài JSON hay khối suy nghĩ/lập luận nào:
```json
{{
  "answer": "Nội dung câu trả lời rõ ràng, súc tích...",
  "citations": [],
  "confidence_level": "FULL | PARTIAL | INSUFFICIENT",
  "abstention_reason": null
}}
```"""

DEFAULT_LLM_ONLY_USER_TEMPLATE_V1 = """CÂU HỎI CỦA NGƯỜI DÙNG:
{query}

Hãy trả lời bằng định dạng JSON chuẩn theo hướng dẫn:"""


class PromptTemplateRegistry:
    """Manages versioned prompt templates for the system."""

    def __init__(self) -> None:
        self._templates: Dict[str, PromptTemplate] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        v1 = PromptTemplate(
            template_version="prompt_v1",
            template_name="standard_legal_qa_v1",
            system_template=DEFAULT_SYSTEM_TEMPLATE_V1,
            user_template=DEFAULT_USER_TEMPLATE_V1,
            instruction_policy_version="v1.0",
        )
        self.register(v1)

        v1_llm_only = PromptTemplate(
            template_version="prompt_llm_only_v1",
            template_name="standard_legal_qa_llm_only_v1",
            system_template=DEFAULT_LLM_ONLY_SYSTEM_TEMPLATE_V1,
            user_template=DEFAULT_LLM_ONLY_USER_TEMPLATE_V1,
            instruction_policy_version="v1.0",
        )
        self.register(v1_llm_only)

    def register(self, template: PromptTemplate) -> None:
        self._templates[template.template_version] = template

    def get(self, template_version: str) -> PromptTemplate:
        if template_version not in self._templates:
            raise InvalidPromptError(
                f"Prompt template version '{template_version}' not registered. "
                f"Available versions: {list(self._templates.keys())}"
            )
        return self._templates[template_version]

    def has_version(self, template_version: str) -> bool:
        return template_version in self._templates
