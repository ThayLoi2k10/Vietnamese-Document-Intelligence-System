# tests/leakage/test_frozen_test_leakage.py
"""Leakage tests bảo vệ tập dữ liệu Frozen Test theo Mục 19 và Mục 22.4.

Tuân thủ:
    - Mục 19.4: Frozen test không được dùng cho tuning/selection/prompting
    - Mục 19.5: Cấu hình tuning trên FROZEN_TEST phải ném EvaluationDataLeakageError
    - Mục 19.7: Khởi tạo pipeline không đòi hỏi nạp frozen test
    - Mục 21.3: ExperimentManifest bắt buộc phải ghi nhận dataset split
"""

import unittest

from src.rag.config.schemas.config_models import SplitRole
from src.rag.config.validation.validator import ConfigValidator
from src.rag.errors.exceptions import EvaluationDataLeakageError
from src.rag.reproducibility.manifest import ExperimentManifest
from tests.fixtures.sample_configs import make_valid_experiment_config


class TestFrozenTestLeakageProtection(unittest.TestCase):
    """Kiểm tra các cổng bảo vệ chống rò rỉ tập dữ liệu frozen test."""

    def test_tuning_on_frozen_test_raises_leakage_error(self) -> None:
        """Kiểm tra hệ thống chặn đứng mọi thực nghiệm tuning cấu hình tập dữ liệu FROZEN_TEST."""
        cfg = make_valid_experiment_config(
            split_role=SplitRole.FROZEN_TEST,
            is_tuning=True,
        )

        with self.assertRaises(EvaluationDataLeakageError) as ctx:
            ConfigValidator.validate(cfg)

        self.assertIn("VI PHẠM RÒ RỈ DỮ LIỆU", str(ctx.exception))
        self.assertIn("FROZEN_TEST", str(ctx.exception))

    def test_manifest_records_dataset_split_role(self) -> None:
        """Kiểm tra ExperimentManifest ghi nhận chính xác vai trò tập dữ liệu (Mục 21.3)."""
        cfg_dev = make_valid_experiment_config(
            split_role=SplitRole.DEVELOPMENT,
            is_tuning=False,
        )
        manifest_dev = ExperimentManifest.from_config(cfg_dev)
        self.assertEqual(
            manifest_dev.dataset["split_role"], SplitRole.DEVELOPMENT.value
        )

        cfg_val = make_valid_experiment_config(
            split_role=SplitRole.VALIDATION,
            is_tuning=False,
        )
        manifest_val = ExperimentManifest.from_config(cfg_val)
        self.assertEqual(
            manifest_val.dataset["split_role"], SplitRole.VALIDATION.value
        )


if __name__ == "__main__":
    unittest.main()
