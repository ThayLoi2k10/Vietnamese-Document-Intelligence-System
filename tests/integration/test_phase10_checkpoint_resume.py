"""Integration tests for experiment runner checkpointing, resumption, and retry logic."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluation.run_llm_only_experiment import (
    load_checkpoint,
    run_llm_only_experiment,
    save_checkpoint,
)


class TestPhase10CheckpointResume(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.checkpoint_path = self.tmp_path / "test_checkpoint.json"
        self.out_dir = self.tmp_path / "out"
        self.art_dir = self.tmp_path / "art"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_checkpoint_save_and_load_roundtrip(self) -> None:
        preds = [{"query_id": 1, "query": "q1", "response": {}}]
        fails = [{"query_id": 2, "query": "q2", "error": "test error"}]

        save_checkpoint(self.checkpoint_path, "exp_test", preds, fails)
        loaded_preds, loaded_fails = load_checkpoint(self.checkpoint_path)

        self.assertEqual(len(loaded_preds), 1)
        self.assertEqual(loaded_preds[0]["query_id"], 1)
        self.assertEqual(len(loaded_fails), 1)
        self.assertEqual(loaded_fails[0]["query_id"], 2)

    def test_resume_skips_completed_queries(self) -> None:
        preds = [
            {
                "query_id": 11525,
                "query": "Biên bản giao nhận tài sản góp vốn...",
                "response": {
                    "query": "Biên bản giao nhận tài sản góp vốn...",
                    "answer": "Nội dung câu trả lời mẫu",
                    "citations": [],
                    "confidence_level": "FULL",
                    "is_abstained": False,
                    "abstention_reason": None,
                    "retrieval_items": [],
                    "model_identifier": "Qwen/Qwen2.5-7B-Instruct",
                    "execution_time_seconds": 0.05,
                    "stage_latencies": {},
                },
            }
        ]
        save_checkpoint(self.checkpoint_path, "exp_phase10_llm_only", preds, [])

        run_llm_only_experiment(
            config_path=Path("configs/experiments/phase10_llm_only_experiment.json"),
            sample_limit_override=2,
            checkpoint_path=self.checkpoint_path,
            checkpoint_freq=1,
            resume=True,
            retry_failed=True,
            dry_run=True,
            output_dir=self.out_dir,
            artifacts_dir=self.art_dir,
        )

        loaded_preds, loaded_fails = load_checkpoint(self.checkpoint_path)
        self.assertEqual(len(loaded_preds), 2)
        q_ids = {p["query_id"] for p in loaded_preds}
        self.assertIn(11525, q_ids)
        self.assertIn(1248, q_ids)
        self.assertEqual(len(loaded_fails), 0)

    def test_retry_failed_queries_recovery(self) -> None:
        fails = [
            {
                "query_id": 11525,
                "query": "Biên bản giao nhận tài sản góp vốn...",
                "error": "MalformedStructuredOutputError: Cú pháp lỗi cũ",
            }
        ]
        save_checkpoint(self.checkpoint_path, "exp_phase10_llm_only", [], fails)

        run_llm_only_experiment(
            config_path=Path("configs/experiments/phase10_llm_only_experiment.json"),
            sample_limit_override=1,
            checkpoint_path=self.checkpoint_path,
            checkpoint_freq=1,
            resume=True,
            retry_failed=True,
            dry_run=True,
            output_dir=self.out_dir,
            artifacts_dir=self.art_dir,
        )

        loaded_preds, loaded_fails = load_checkpoint(self.checkpoint_path)
        self.assertEqual(len(loaded_preds), 1)
        self.assertEqual(loaded_preds[0]["query_id"], 11525)
        self.assertEqual(len(loaded_fails), 0)


if __name__ == "__main__":
    unittest.main()
