# tests/unit/config/test_config.py
"""Unit tests cho module Configuration & Validation Phase 9.

Tuân thủ Mục 22.2: Configuration unit tests:
    - Missing required fields
    - Incompatible component combinations (depth hierarchy)
    - Invalid context budget configuration
    - Frozen-test tuning rejection (Leakage prevention)
"""

import unittest

from src.rag.config.schemas.config_models import SplitRole
from src.rag.config.validation.validator import ConfigValidator
from src.rag.errors.exceptions import (
    ConfigurationValidationError,
    EvaluationDataLeakageError,
)
from tests.fixtures.sample_configs import make_valid_experiment_config


class TestConfigModule(unittest.TestCase):
    """Kiểm tra tính hợp lệ của cấu hình và các bất biến kiến trúc."""

    def test_valid_config_passes(self) -> None:
        """Kiểm tra cấu hình hợp lệ đi qua validator thành công."""
        cfg = make_valid_experiment_config()
        ConfigValidator.validate(cfg)  # Không ném ngoại lệ

    def test_incompatible_depths_retrieval_less_than_context(self) -> None:
        """Kiểm tra báo lỗi khi retrieval_top_k < context_top_k."""
        cfg = make_valid_experiment_config()
        cfg.retriever.retrieval_top_k = 3
        cfg.context.context_top_k = 5  # Lớn hơn retrieval_top_k

        with self.assertRaises(ConfigurationValidationError):
            ConfigValidator.validate(cfg)

    def test_incompatible_depths_with_reranker(self) -> None:
        """Kiểm tra phân cấp độ sâu khi bật reranker (retrieval_top_k >= reranker_top_k >= context_top_k)."""
        cfg = make_valid_experiment_config(reranker_enabled=True)
        cfg.retriever.retrieval_top_k = 10
        cfg.reranker.reranker_top_k = 15  # Lớn hơn retrieval_top_k -> Lỗi!

        with self.assertRaises(ConfigurationValidationError):
            ConfigValidator.validate(cfg)

    def test_invalid_budget(self) -> None:
        """Kiểm tra ngân sách token âm hoặc bằng 0."""
        cfg = make_valid_experiment_config()
        cfg.context.max_context_tokens = 0

        with self.assertRaises(ConfigurationValidationError):
            ConfigValidator.validate(cfg)

    def test_frozen_test_tuning_rejection_raises_leakage_error(self) -> None:
        """Kiểm tra ném EvaluationDataLeakageError khi cấu hình tuning trên FROZEN_TEST (Mục 19.5)."""
        cfg = make_valid_experiment_config(
            split_role=SplitRole.FROZEN_TEST,
            is_tuning=True,  # Vi phạm: không được tuning trên tập FROZEN_TEST
        )
        with self.assertRaises(EvaluationDataLeakageError):
            ConfigValidator.validate(cfg)


if __name__ == "__main__":
    unittest.main()
