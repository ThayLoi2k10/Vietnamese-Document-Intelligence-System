# src/rag/config/validation/validator.py
"""Bộ kiểm định tính hợp lệ và chống rò rỉ dữ liệu cấu hình thực nghiệm.

Tuân thủ:
    - Mục 18.3: Cross-component compatibility validation
    - Mục 19.4 & 19.5: Leakage protection (Từ chối cấu hình tuning trên FROZEN_TEST)
"""

from src.rag.config.schemas.config_models import ExperimentConfig, SplitRole
from src.rag.errors.exceptions import (
    ConfigurationValidationError,
    EvaluationDataLeakageError,
)


class ConfigValidator:
    """Xác thực toàn diện các ràng buộc và bất biến của ExperimentConfig."""

    @staticmethod
    def validate(config: ExperimentConfig) -> None:
        # 1. RÀNG BUỘC CHỐNG RÒ RỈ DỮ LIỆU KIỂM THỬ (LEAKAGE PROTECTION - MỤC 19.5)
        if config.dataset.split_role == SplitRole.FROZEN_TEST:
            if config.is_tuning_experiment:
                raise EvaluationDataLeakageError(
                    f"VI PHẠM RÒ RỈ DỮ LIỆU: Thực nghiệm '{config.experiment_id}' được đánh dấu là "
                    f"tuning/selection nhưng lại sử dụng tập dữ liệu FROZEN_TEST! "
                    f"Tập FROZEN_TEST chỉ được phép dùng cho kiểm định cuối sau khi đóng băng kiến trúc.",
                    experiment_id=config.experiment_id,
                    component_identifier="ConfigValidator",
                )

        # 2. KIỂM TRA ĐỘ SÂU PHÂN CẤP (SEPARATION OF DEPTHS - MỤC 5.4 & 9.5)
        ret_k = config.retriever.retrieval_top_k
        ctx_k = config.context.context_top_k

        if ret_k <= 0:
            raise ConfigurationValidationError(
                f"retrieval_top_k phải > 0, nhận được: {ret_k}"
            )
        if ctx_k <= 0:
            raise ConfigurationValidationError(
                f"context_top_k phải > 0, nhận được: {ctx_k}"
            )

        if config.reranker.enabled:
            rerank_k = config.reranker.reranker_top_k
            if rerank_k <= 0:
                raise ConfigurationValidationError(
                    f"reranker_top_k phải > 0 khi reranker bật, nhận được: {rerank_k}"
                )
            if ret_k < rerank_k:
                raise ConfigurationValidationError(
                    f"Bất hợp lý: retrieval_top_k ({ret_k}) < reranker_top_k ({rerank_k}). "
                    f"Số ứng viên truyền cho reranker không thể lớn hơn số ứng viên truy xuất."
                )
            if rerank_k < ctx_k:
                raise ConfigurationValidationError(
                    f"Bất hợp lý: reranker_top_k ({rerank_k}) < context_top_k ({ctx_k}). "
                    f"Số ứng viên chọn vào context không thể lớn hơn số ứng viên đã rerank."
                )
        else:
            if ret_k < ctx_k:
                raise ConfigurationValidationError(
                    f"Bất hợp lý: retrieval_top_k ({ret_k}) < context_top_k ({ctx_k}). "
                    f"Không thể chọn nhiều context item hơn số lượng tài liệu đã truy xuất."
                )

        # 3. KIỂM TRA NGÂN SÁCH TOKEN (TOKEN BUDGET - MỤC 10.8)
        if config.context.max_context_tokens <= 0:
            raise ConfigurationValidationError(
                f"max_context_tokens phải > 0, nhận được: {config.context.max_context_tokens}"
            )
        if config.generation.max_output_tokens <= 0:
            raise ConfigurationValidationError(
                f"max_output_tokens phải > 0, nhận được: {config.generation.max_output_tokens}"
            )

        # 4. KIỂM TRA CÁC TRƯỜNG ĐỊNH DANH BẮT BUỘC
        if not config.experiment_id or not config.experiment_id.strip():
            raise ConfigurationValidationError("experiment_id không được rỗng.")
        if not config.generation.llm_model or not config.generation.llm_model.strip():
            raise ConfigurationValidationError("llm_model không được rỗng.")
        if not config.prompting.prompt_version or not config.prompting.prompt_version.strip():
            raise ConfigurationValidationError("prompt_version không được rỗng.")
