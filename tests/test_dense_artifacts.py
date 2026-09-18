# tests/test_dense_artifacts.py
"""Unit tests cho DenseArtifactManager, bảo toàn tính toàn vẹn fingerprint và fail-fast."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np

from src.retrieval.dense.artifacts import (
    DenseArtifactManager,
    compute_corpus_fingerprint,
    compute_id_sequence_fingerprint,
)


class TestDenseArtifacts(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.manager = DenseArtifactManager(self.temp_dir)
        self.count = 10
        self.dim = 1024

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fingerprint_computation(self):
        records1 = [(0, "Điều 1"), (1, "Điều 2")]
        records2 = [(0, "Điều 1"), (1, "Điều 2")]
        records3 = [(1, "Điều 2"), (0, "Điều 1")]  # Khác thứ tự

        fp1 = compute_corpus_fingerprint(records1)
        fp2 = compute_corpus_fingerprint(records2)
        fp3 = compute_corpus_fingerprint(records3)

        self.assertEqual(fp1, fp2)
        self.assertNotEqual(fp1, fp3)  # Thứ tự khác nhau tạo ra hash khác nhau

    def test_save_and_verify_success(self):
        config = {"model": {"name": "BAAI/bge-m3"}}
        corpus_ids = list(range(self.count))
        corpus_fp = {"corpus_text_id_sha256": "dummy_hash_123"}
        embeddings = np.ones((self.count, self.dim), dtype=np.float32)

        def mock_faiss_saver(path):
            with open(path, "w") as f:
                f.write("mock faiss content")

        build_report = {"elapsed": 10.0}

        self.manager.save_all_artifacts(
            config=config,
            corpus_ids=corpus_ids,
            corpus_fingerprint=corpus_fp,
            embeddings=embeddings,
            faiss_index_saver=mock_faiss_saver,
            build_report=build_report,
        )

        # Kiểm tra verification
        res = self.manager.verify_artifacts(
            expected_corpus_fingerprint="dummy_hash_123",
            expected_count=self.count,
        )
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["corpus_ids_count"], self.count)
        self.assertEqual(res["embeddings_shape"], [self.count, self.dim])

    def test_verify_fail_fast_fingerprint_mismatch(self):
        config = {"model": {"name": "BAAI/bge-m3"}}
        corpus_ids = list(range(self.count))
        corpus_fp = {"corpus_text_id_sha256": "hash_A"}
        embeddings = np.ones((self.count, self.dim), dtype=np.float32)

        def mock_faiss_saver(path):
            with open(path, "w") as f:
                f.write("mock")

        self.manager.save_all_artifacts(
            config=config,
            corpus_ids=corpus_ids,
            corpus_fingerprint=corpus_fp,
            embeddings=embeddings,
            faiss_index_saver=mock_faiss_saver,
            build_report={},
        )

        with self.assertRaises(ValueError):
            self.manager.verify_artifacts(
                expected_corpus_fingerprint="hash_B",
                expected_count=self.count,
            )

    def test_verify_fail_fast_missing_file(self):
        # Chưa lưu gì mà gọi verify
        with self.assertRaises(FileNotFoundError):
            self.manager.verify_artifacts(expected_count=self.count)


if __name__ == "__main__":
    unittest.main()
