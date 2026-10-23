"""Unit tests for LegalPassageCompressor extractive context compression."""

import unittest

from src.rag.context.compressor import CompressedPassage, LegalPassageCompressor


class TestLegalPassageCompressor(unittest.TestCase):
    def setUp(self) -> None:
        self.compressor = LegalPassageCompressor(
            max_chars_per_article=1200,
            min_chars_to_compress=600,
        )
        self.long_article = (
            "Điều 5. Tiêu chuẩn tuyển chọn công dân thực hiện nghĩa vụ tham gia Công an nhân dân\n\n"
            "1. Có quốc tịch là quốc tịch Việt Nam và cư trú tại Việt Nam.\n\n"
            "2. Từ đủ 18 tuổi đến hết 25 tuổi; công dân có trình độ cao đẳng, đại học đã được tạm hoãn gọi nhập ngũ "
            "thì độ tuổi gọi nhập ngũ đến hết 27 tuổi đối với các trường hợp theo quy định.\n\n"
            "3. Có lý lịch rõ ràng. Nghiêm chỉnh chấp hành đường lối, chủ trương của Đảng, chính sách, pháp luật của Nhà nước; "
            "không có tiền án, tiền sự, không bị truy cứu trách nhiệm hình sự, quản chế, không trong thời gian bị áp dụng biện pháp giáo dục tại xã, phường, thị trấn.\n\n"
            "4. Đủ sức khỏe phục vụ tại ngũ theo kết luận của Hội đồng khám sức khỏe theo quy định của Bộ Y tế và Bộ Công an.\n\n"
            "5. Có bằng tốt nghiệp trung học phổ thông trở lên. Các xã miền núi, hải đảo, vùng đồng bào dân tộc thiểu số được tuyển công dân tốt nghiệp trung học cơ sở.\n\n"
            "6. Đáp ứng các tiêu chuẩn về năng khiếu, phẩm chất chính trị theo quy định cụ thể của Bộ Công an trong từng thời kỳ tuyển chọn tuyển dụng."
        )

    def test_short_article_retained(self) -> None:
        short_text = "Điều 26. Tiền lương trong thời gian thử việc\nTiền lương ít nhất bằng 85% lương chính thức."
        res = self.compressor.compress_article("tiền lương thử việc", short_text)

        self.assertFalse(res.is_compressed)
        self.assertEqual(res.compressed_text, short_text)
        self.assertEqual(res.compression_ratio, 1.0)
        self.assertEqual(res.clauses_retained, ["toàn_văn"])

    def test_extractive_clause_focus(self) -> None:
        query = "người có tiền án tiền sự có được vào công an không"
        res = self.compressor.compress_article(query, self.long_article)

        self.assertTrue(res.is_compressed)
        self.assertLess(res.compressed_length, res.original_length)
        self.assertIn("Điều 5. Tiêu chuẩn tuyển chọn", res.compressed_text)
        self.assertIn("không có tiền án, tiền sự", res.compressed_text)
        self.assertIn("Khoản 3", res.clauses_retained)
        self.assertIn("Tiêu đề", res.clauses_retained)

    def test_multi_clause_selection_chronological(self) -> None:
        query = "độ tuổi và sức khỏe tuyển chọn công an"
        res = self.compressor.compress_article(query, self.long_article)

        self.assertTrue(res.is_compressed)
        self.assertIn("Từ đủ 18 tuổi đến hết 25 tuổi", res.compressed_text)
        self.assertIn("Đủ sức khỏe phục vụ", res.compressed_text)
        idx_age = res.compressed_text.find("Từ đủ 18 tuổi")
        idx_health = res.compressed_text.find("Đủ sức khỏe")
        self.assertLess(idx_age, idx_health, "Các khoản phải được giữ theo đúng thứ tự thời gian gốc")

    def test_compress_documents(self) -> None:
        docs = [
            {
                "rank": 1,
                "law_id": "70/2019/NĐ-CP",
                "aid": "5",
                "score": 0.95,
                "text": self.long_article,
            },
            {
                "rank": 2,
                "law_id": "45/2019/QH14",
                "aid": "26",
                "score": 0.50,
                "text": "Điều 26. Tiền lương thử việc ít nhất 85%.",
            },
        ]
        query = "tiền án tiền sự công an"
        comp_docs = self.compressor.compress_documents(query, docs)

        self.assertEqual(len(comp_docs), 2)
        self.assertTrue(comp_docs[0]["is_compressed"])
        self.assertIn("raw_text", comp_docs[0])
        self.assertEqual(comp_docs[0]["raw_text"], self.long_article)
        self.assertFalse(comp_docs[1]["is_compressed"])

    def test_fallback_when_no_direct_keyword_match(self) -> None:
        query = "một câu hỏi hoàn toàn xa lạ không có chữ nào trong điều luật"
        res = self.compressor.compress_article(query, self.long_article)

        self.assertTrue(res.is_compressed)
        self.assertGreater(len(res.compressed_text), 0)
        self.assertIn("Điều 5. Tiêu chuẩn tuyển chọn", res.compressed_text)

    def test_empty_input(self) -> None:
        res = self.compressor.compress_article("câu hỏi", "")
        self.assertFalse(res.is_compressed)
        self.assertEqual(res.compressed_text, "")
        self.assertEqual(res.original_length, 0)


if __name__ == "__main__":
    unittest.main()
