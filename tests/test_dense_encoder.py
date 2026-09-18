# tests/test_dense_encoder.py
"""Unit tests cho DenseEncoder với mô hình BAAI/bge-m3."""

import unittest
import numpy as np

from src.retrieval.dense.encoder import (
    EMBEDDING_DIMENSION,
    DenseEncoder,
)


class TestDenseEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Khởi tạo encoder một lần cho toàn bộ test suite để tiết kiệm thời gian nạp mô hình
        cls.encoder = DenseEncoder()

    def test_encode_queries_shape_and_dtype(self):
        queries = ["Hành vi vượt đèn đỏ bị xử phạt như thế nào?", "Quy định về bảo hiểm y tế."]
        vecs = self.encoder.encode_queries(queries, batch_size=2)

        self.assertIsInstance(vecs, np.ndarray)
        self.assertEqual(vecs.shape, (2, EMBEDDING_DIMENSION))
        self.assertEqual(vecs.dtype, np.float32)

        # Kiểm tra chuẩn L2 xấp xỉ 1.0
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], rtol=1e-3, atol=1e-3)

    def test_encode_empty_input_list(self):
        vecs = self.encoder.encode_queries([])
        self.assertEqual(vecs.shape, (0, EMBEDDING_DIMENSION))
        self.assertEqual(vecs.dtype, np.float32)

        vecs_art = self.encoder.encode_articles([])
        self.assertEqual(vecs_art.shape, (0, EMBEDDING_DIMENSION))

    def test_empty_article_handling(self):
        """Kiểm tra tài liệu rỗng hoặc khoảng trắng vẫn trả về vector hợp lệ mà không bị loại bỏ."""
        articles = ["", "   ", "Điều 1. Nội dung bình thường."]
        vecs = self.encoder.encode_articles(articles, batch_size=3)

        self.assertEqual(vecs.shape, (3, EMBEDDING_DIMENSION))
        self.assertEqual(vecs.dtype, np.float32)

        # Vector không được là zero-vector
        norms = np.linalg.norm(vecs, axis=1)
        for n in norms:
            self.assertGreater(n, 0.5)

    def test_determinism(self):
        """Kiểm tra tính tất định: cùng văn bản trong cùng môi trường sinh ra cùng vector."""
        query = "Thủ tục đăng ký kinh doanh doanh nghiệp tư nhân."
        v1 = self.encoder.encode_queries([query])
        v2 = self.encoder.encode_queries([query])

        np.testing.assert_allclose(v1, v2, rtol=1e-5, atol=1e-5)

    def test_zero_vector_rejection(self):
        """Kiểm tra phương thức kiểm định bắt lỗi zero vector (Fail-fast)."""
        bad_matrix = np.zeros((2, EMBEDDING_DIMENSION), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.encoder._validate_embeddings(bad_matrix, expected_count=2)


if __name__ == "__main__":
    unittest.main()
