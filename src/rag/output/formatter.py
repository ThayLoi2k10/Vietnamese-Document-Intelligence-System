"""Structured legal output formatter and section parser for legal RAG responses."""

import re
from typing import Any, Dict, List, Optional


class LegalResponseFormatter:
    """Format and parse structured legal assistant answers."""

    SYSTEM_PROMPT_TEMPLATE = (
        "Bạn là một Luật sư và Chuyên gia Pháp lý AI của Việt Nam. "
        "Hãy dựa vào các văn bản luật được cung cấp dưới đây để trả lời câu hỏi của người dùng theo ĐÚNG CẤU TRÚC 3 PHẦN sau:\n\n"
        "### 🎯 1. KẾT LUẬN TRỰC DIỆN\n"
        "- Trả lời thẳng vào câu hỏi một cách ngắn gọn, rõ ràng: nêu rõ có/không, số ngày, mốc thời gian, mức tiền hoặc điều kiện cụ thể.\n\n"
        "### ⚖️ 2. CĂN CỨ PHÁP LÝ\n"
        "- Nêu rõ tên văn bản quy phạm pháp luật và điều khoản cụ thể làm căn cứ (ví dụ: Khoản 1 Điều 25 Bộ luật Lao động 2019).\n"
        "- Trích dẫn ngắn gọn câu chữ quy định trong điều luật làm bằng chứng xác thực.\n\n"
        "### 📋 3. LƯU Ý & ĐIỀU KIỆN ÁP DỤNG\n"
        "- Nêu các trường hợp ngoại lệ, đối tượng loại trừ, hoặc các thủ tục cần thiết mà người dùng cần chú ý.\n\n"
        "QUY TẮC BẮT BUỘC:\n"
        "- Nếu trong các điều luật được cung cấp KHÔNG có đủ thông tin để giải đáp câu hỏi, hãy ghi rõ:\n"
        "  '### ⚠️ THÔNG BÁO THIẾU CĂN CỨ\n"
        "  Dựa trên các văn bản luật được cung cấp, không có đủ căn cứ pháp lý để giải đáp câu hỏi này.'\n"
        "- Tuyệt đối KHÔNG tự ý suy diễn hoặc bịa đặt điều luật không có trong tài liệu được cung cấp."
    )

    @classmethod
    def parse_sections(cls, text: str) -> Dict[str, Any]:
        """Parse structured 3-part legal response into a normalized dictionary."""
        if not text or not text.strip():
            return {
                "direct_conclusion": "",
                "legal_grounds": "",
                "notes_and_conditions": "",
                "is_abstention": False,
                "raw_text": text,
            }

        is_abstention = (
            "THÔNG BÁO THIẾU CĂN CỨ" in text
            or "không có đủ căn cứ" in text.lower()
            or "không có đủ thông tin" in text.lower()
        )

        conclusion_match = re.search(
            r"###\s*🎯?\s*1\.?\s*KẾT\s+LUẬN[^\n]*\n(.*?)(?=(###|\Z))",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        grounds_match = re.search(
            r"###\s*⚖️?\s*2\.?\s*CĂN\s+CỨ[^\n]*\n(.*?)(?=(###|\Z))",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        notes_match = re.search(
            r"###\s*📋?\s*3\.?\s*LƯU\s+Ý[^\n]*\n(.*?)(?=(###|\Z))",
            text,
            re.DOTALL | re.IGNORECASE,
        )

        return {
            "direct_conclusion": (
                conclusion_match.group(1).strip() if conclusion_match else ""
            ),
            "legal_grounds": grounds_match.group(1).strip() if grounds_match else "",
            "notes_and_conditions": notes_match.group(1).strip() if notes_match else "",
            "is_abstention": is_abstention,
            "raw_text": text,
        }

    @classmethod
    def to_api_json(
        cls,
        text: str,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Convert parsed legal answer and source docs to production API schema."""
        parsed = cls.parse_sections(text)
        sources = []
        if retrieved_docs:
            for d in retrieved_docs:
                sources.append(
                    {
                        "rank": d.get("rank"),
                        "law_id": d.get("law_id"),
                        "article_id": d.get("aid") or d.get("article_id"),
                        "score": d.get("score"),
                    }
                )

        from src.rag.citation.grounding import GroundingVerifier

        verifier = GroundingVerifier()
        grounding_result = verifier.verify(text, retrieved_docs)

        return {
            "status": "ABSTAINED" if parsed["is_abstention"] else "SUCCESS",
            "direct_conclusion": parsed["direct_conclusion"],
            "legal_grounds": parsed["legal_grounds"],
            "notes_and_conditions": parsed["notes_and_conditions"],
            "grounding": grounding_result.to_dict(),
            "sources": sources,
            "full_answer": parsed["raw_text"],
        }
