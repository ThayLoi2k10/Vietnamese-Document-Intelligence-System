"""Unit tests for structured legal response formatter."""

import unittest
from src.rag.output.formatter import LegalResponseFormatter


class TestLegalResponseFormatter(unittest.TestCase):
    def test_parse_full_structured_response(self):
        sample_output = """
### 🎯 1. KẾT LUẬN TRỰC DIỆN
Thời gian thử việc tối đa là 60 ngày đối với công việc có trình độ cao đẳng trở lên, và 180 ngày đối với người quản lý doanh nghiệp.

### ⚖️ 2. CĂN CỨ PHÁP LÝ
- Căn cứ Điều 25 Bộ luật Lao động 2019 (văn bản 45/2019/QH14).
- Trích dẫn: "Thời gian thử việc do hai bên thỏa thuận căn cứ vào tính chất và mức độ phức tạp của công việc..."

### 📋 3. LƯU Ý & ĐIỀU KIỆN ÁP DỤNG
- Chỉ được thử việc 01 lần đối với một công việc.
- Không áp dụng thử việc đối với người lao động giao kết hợp đồng lao động có thời hạn dưới 01 tháng.
"""
        parsed = LegalResponseFormatter.parse_sections(sample_output)

        self.assertFalse(parsed["is_abstention"])
        self.assertIn("60 ngày", parsed["direct_conclusion"])
        self.assertIn("Điều 25 Bộ luật Lao động 2019", parsed["legal_grounds"])
        self.assertIn("Chỉ được thử việc 01 lần", parsed["notes_and_conditions"])

    def test_parse_abstention(self):
        sample_abstention = """
### ⚠️ THÔNG BÁO THIẾU CĂN CỨ
Dựa trên các văn bản luật được cung cấp, không có đủ căn cứ pháp lý để giải đáp câu hỏi này.
"""
        parsed = LegalResponseFormatter.parse_sections(sample_abstention)
        self.assertTrue(parsed["is_abstention"])

    def test_to_api_json(self):
        text = """
### 🎯 1. KẾT LUẬN TRỰC DIỆN
Báo trước ít nhất 45 ngày.

### ⚖️ 2. CĂN CỨ PHÁP LÝ
Điều 35 Bộ luật Lao động 2019.

### 📋 3. LƯU Ý & ĐIỀU KIỆN ÁP DỤNG
Áp dụng cho HĐLĐ không xác định thời hạn.
"""
        docs = [
            {"rank": 1, "law_id": "45/2019/QH14", "aid": "54231", "score": 0.05}
        ]
        api_data = LegalResponseFormatter.to_api_json(text, retrieved_docs=docs)

        self.assertEqual(api_data["status"], "SUCCESS")
        self.assertIn("45 ngày", api_data["direct_conclusion"])
        self.assertEqual(len(api_data["sources"]), 1)
        self.assertEqual(api_data["sources"][0]["law_id"], "45/2019/QH14")

    def test_empty_string(self):
        parsed = LegalResponseFormatter.parse_sections("")
        self.assertEqual(parsed["direct_conclusion"], "")
        self.assertFalse(parsed["is_abstention"])


if __name__ == "__main__":
    unittest.main()
