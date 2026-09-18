# tests/test_dense_index.py
"""Unit tests cho DenseIndex (FAISS IndexFlatIP) và kiểm định tính chính xác so với NumPy oracle."""

import shutil
import tempfile
import unittest
import numpy as np

from src.retrieval.dense.index import DenseIndex


class TestDenseIndex(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dimension = 1024
        self.num_docs = 20

        # Tạo ma trận ngẫu nhiên và chuẩn hóa L2
        rng = np.random.default_rng(42)
        raw_vecs = rng.standard_normal((self.num_docs, self.dimension)).astype(np.float32)
        norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        self.doc_embeddings = raw_vecs / norms
        self.corpus_ids = [1000 + i for i in range(self.num_docs)]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_and_properties(self):
        index = DenseIndex(dimension=self.dimension)
        index.add(self.doc_embeddings, self.corpus_ids)

        self.assertEqual(index.ntotal, self.num_docs)
        self.assertEqual(index.dimension, self.dimension)
        self.assertEqual(len(index.corpus_ids), self.num_docs)
        self.assertEqual(index.corpus_ids, self.corpus_ids)

    def test_search_score_ordering_and_mapping(self):
        index = DenseIndex(dimension=self.dimension)
        index.add(self.doc_embeddings, self.corpus_ids)

        # Truy vấn bằng chính doc 0 -> Điểm cao nhất phải xấp xỉ 1.0 và trả về ID của doc 0
        q_vec = self.doc_embeddings[0:1]
        scores, doc_ids = index.search(q_vec, top_k=5)

        self.assertEqual(len(doc_ids[0]), 5)
        self.assertEqual(doc_ids[0][0], self.corpus_ids[0])
        self.assertAlmostEqual(scores[0][0], 1.0, places=5)

        # Điểm số phải giảm dần
        for i in range(len(scores[0]) - 1):
            self.assertGreaterEqual(scores[0][i], scores[0][i + 1])

    def test_exactness_vs_numpy_oracle(self):
        """Kiểm tra độ chính xác xếp hạng giữa FAISS IndexFlatIP và NumPy dot product."""
        index = DenseIndex(dimension=self.dimension)
        index.add(self.doc_embeddings, self.corpus_ids)

        rng = np.random.default_rng(123)
        q_raw = rng.standard_normal((3, self.dimension)).astype(np.float32)
        q_norms = np.linalg.norm(q_raw, axis=1, keepdims=True)
        query_embeddings = q_raw / q_norms

        # NumPy oracle: Q @ D.T
        numpy_sims = query_embeddings @ self.doc_embeddings.T

        top_k = 10
        faiss_scores, faiss_ids = index.search(query_embeddings, top_k=top_k)

        for q_idx in range(3):
            # Thứ hạng kỳ vọng từ NumPy
            expected_ranked_indices = np.argsort(-numpy_sims[q_idx])[:top_k]
            expected_ranked_ids = [self.corpus_ids[idx] for idx in expected_ranked_indices]
            expected_scores = numpy_sims[q_idx][expected_ranked_indices]

            # Kiểm tra ID khớp hoàn toàn
            self.assertEqual(faiss_ids[q_idx], expected_ranked_ids)

            # Kiểm tra điểm số khớp trong dung sai số thực
            np.testing.assert_allclose(
                faiss_scores[q_idx], expected_scores, rtol=1e-5, atol=1e-5
            )

    def test_edge_cases_top_k(self):
        index = DenseIndex(dimension=self.dimension)
        index.add(self.doc_embeddings, self.corpus_ids)

        # top_k lớn hơn tổng số tài liệu
        scores, doc_ids = index.search(self.doc_embeddings[0:1], top_k=100)
        self.assertEqual(len(doc_ids[0]), self.num_docs)

        # top_k <= 0
        with self.assertRaises(ValueError):
            index.search(self.doc_embeddings[0:1], top_k=0)
        with self.assertRaises(ValueError):
            index.search(self.doc_embeddings[0:1], top_k=-5)

    def test_save_and_load(self):
        index = DenseIndex(dimension=self.dimension)
        index.add(self.doc_embeddings, self.corpus_ids)

        index_file = f"{self.temp_dir}/test_index.faiss"
        ids_file = f"{self.temp_dir}/test_ids.json"

        index.save(index_file, ids_file)

        loaded_index = DenseIndex.load(index_file, ids_file)
        self.assertEqual(loaded_index.ntotal, self.num_docs)
        self.assertEqual(loaded_index.corpus_ids, self.corpus_ids)

        # Kết quả tìm kiếm trước và sau khi load phải đồng nhất
        scores_orig, ids_orig = index.search(self.doc_embeddings[0:2], top_k=5)
        scores_load, ids_load = loaded_index.search(self.doc_embeddings[0:2], top_k=5)

        self.assertEqual(ids_orig, ids_load)
        np.testing.assert_allclose(scores_orig, scores_load, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
