"""Unit tests for GroundingVerifier and citation provenance cross-referencing."""

import unittest

from src.rag.citation.grounding import GroundingResult, GroundingVerifier


class TestGroundingVerifier(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier = GroundingVerifier()
        self.retrieved_docs = [
            {
                "rank": 1,
                "article_id": 54223,
                "law_id": "45/2019/QH14",
                "aid": "25",
                "score": 0.88,
                "text": (
                    "Điều 25. Thời gian thử việc\n"
                    "Thời gian thử việc do hai bên thỏa thuận căn cứ vào tính chất và mức độ phức tạp của công việc "
                    "nhưng chỉ được thử việc một lần đối với một công việc và bảo đảm điều kiện sau đây:\n"
                    "1. Không quá 180 ngày đối với công việc của người quản lý doanh nghiệp.\n"
                    "2. Không quá 60 ngày đối với công việc có chức danh nghề nghiệp cần trình độ chuyên môn, kỹ thuật từ cao đẳng trở lên."
                ),
            },
            {
                "rank": 2,
                "article_id": 54224,
                "law_id": "45/2019/QH14",
                "aid": "26",
                "score": 0.81,
                "text": (
                    "Điều 26. Tiền lương trong thời gian thử việc\n"
                    "Tiền lương của người lao động trong thời gian thử việc do hai bên thỏa thuận nhưng ít nhất phải bằng 85% mức lương của công việc đó."
                ),
            },
        ]

    def test_perfect_grounded_response(self) -> None:
        response = (
            "### 🎯 1. KẾT LUẬN TRỰC DIỆN\n"
            "Thời gian thử việc tối đa là 60 ngày đối với công việc đòi hỏi trình độ từ cao đẳng trở lên.\n\n"
            "### ⚖️ 2. CĂN CỨ PHÁP LÝ\n"
            "Căn cứ theo Điều 25 Bộ luật Lao động (Văn bản số 45/2019/QH14):\n"
            '\"Không quá 60 ngày đối với công việc có chức danh nghề nghiệp cần trình độ chuyên môn, kỹ thuật từ cao đẳng trở lên.\"\n\n'
            "### 📋 3. LƯU Ý & ĐIỀU KIỆN ÁP DỤNG\n"
            "Theo Điều 26, tiền lương thử việc ít nhất bằng 85% mức lương của công việc đó."
        )
        res = self.verifier.verify(response, self.retrieved_docs)

        self.assertTrue(res.is_grounded)
        self.assertEqual(res.verdict, "VERIFIED")
        self.assertGreaterEqual(res.grounding_score, 0.9)
        self.assertIn("45/2019/QH14", res.verified_laws)
        self.assertIn("Điều 25", res.verified_articles)
        self.assertIn("Điều 26", res.verified_articles)
        self.assertEqual(len(res.unverified_laws), 0)
        self.assertEqual(len(res.unverified_articles), 0)
        self.assertTrue(res.quote_verifications[0]["matched"])
        self.assertEqual(res.quote_verifications[0]["matched_doc_rank"], 1)

    def test_hallucinated_citations(self) -> None:
        response = (
            "Theo quy định tại Điều 300 Bộ luật Lao động và Nghị định 999/2025/NĐ-CP, "
            "thời gian thử việc được kéo dài tối đa 1 năm."
        )
        res = self.verifier.verify(response, self.retrieved_docs)

        self.assertFalse(res.is_grounded)
        self.assertEqual(res.verdict, "HALLUCINATION_RISK")
        self.assertIn("999/2025/NĐ-CP", res.unverified_laws)
        self.assertIn("Điều 300", res.unverified_articles)
        self.assertIn("CẢNH BÁO ẢO GIÁC", res.badge)

    def test_fabricated_quote_detection(self) -> None:
        response = (
            "Theo Điều 25 số 45/2019/QH14: "
            '\"Người lao động được quyền nghỉ việc không cần thông báo trước trong thời gian thử việc.\"'
        )
        res = self.verifier.verify(response, self.retrieved_docs)

        self.assertEqual(len(res.quote_verifications), 1)
        self.assertFalse(res.quote_verifications[0]["matched"])
        self.assertIsNone(res.quote_verifications[0]["matched_doc_rank"])

    def test_abstention_handling(self) -> None:
        response = (
            "### ⚠️ THÔNG BÁO THIẾU CĂN CỨ\n"
            "Dựa trên các văn bản luật được cung cấp, không có đủ căn cứ pháp lý để giải đáp câu hỏi này."
        )
        res = self.verifier.verify(response, self.retrieved_docs)

        self.assertTrue(res.is_grounded)
        self.assertEqual(res.verdict, "ABSTAINED")
        self.assertEqual(res.grounding_score, 1.0)
        self.assertIn("TỪ CHỐI DO THIẾU CĂN CỨ", res.badge)

    def test_empty_response(self) -> None:
        res = self.verifier.verify("", self.retrieved_docs)
        self.assertTrue(res.is_grounded)
        self.assertEqual(res.verdict, "ABSTAINED")

    def test_to_dict_structure(self) -> None:
        response = "Căn cứ Điều 25 Văn bản 45/2019/QH14."
        res = self.verifier.verify(response, self.retrieved_docs)
        d = res.to_dict()

        self.assertIn("is_grounded", d)
        self.assertIn("verdict", d)
        self.assertIn("grounding_score", d)
        self.assertIn("badge", d)
        self.assertIn("verified_laws", d)
        self.assertIn("verified_articles", d)


if __name__ == "__main__":
    unittest.main()
