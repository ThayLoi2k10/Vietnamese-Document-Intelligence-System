# src/rag/errors/exceptions.py
"""Đặc tả phân cấp ngoại lệ tường minh (Error Taxonomy) cho Phase 9 RAG.

Tuân thủ Mục 20: Error Handling:
    - Typed, explicit, attributable to pipeline stage
    - Preserved in experiment logs, non-silent
    - Section 20.2: 12 error classes chuẩn
    - Section 20.5: Error provenance (experiment_id, pipeline_stage, component_identifier, details)
"""

from typing import Any, Dict, Optional


class RAGError(Exception):
    """Lớp ngoại lệ cơ sở cho toàn bộ hệ thống RAG Phase 9."""

    def __init__(
        self,
        message: str,
        experiment_id: Optional[str] = None,
        pipeline_stage: Optional[str] = None,
        component_identifier: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.experiment_id = experiment_id
        self.pipeline_stage = pipeline_stage
        self.component_identifier = component_identifier
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "experiment_id": self.experiment_id,
            "pipeline_stage": self.pipeline_stage,
            "component_identifier": self.component_identifier,
            "details": self.details,
        }


class InvalidRetrievalResultError(RAGError):
    """Ngoại lệ khi kết quả truy xuất không đúng quy chuẩn (missing identity, invalid rank, v.v.)."""

    def __init__(
        self,
        message: str,
        experiment_id: Optional[str] = None,
        pipeline_stage: str = "retrieval",
        component_identifier: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            experiment_id=experiment_id,
            pipeline_stage=pipeline_stage,
            component_identifier=component_identifier,
            details=details,
        )


class MissingRequiredMetadataError(RAGError):
    """Ngoại lệ khi thiếu metadata bắt buộc để xác định nguồn gốc pháp lý (law_id, chunk_id, v.v.)."""

    def __init__(
        self,
        message: str,
        experiment_id: Optional[str] = None,
        pipeline_stage: Optional[str] = None,
        component_identifier: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            experiment_id=experiment_id,
            pipeline_stage=pipeline_stage,
            component_identifier=component_identifier,
            details=details,
        )


class EmptyRetrievalError(RAGError):
    """Ngoại lệ khi truy xuất rỗng và chính sách cấu hình yêu cầu leo thang thành lỗi."""

    def __init__(
        self,
        message: str,
        experiment_id: Optional[str] = None,
        pipeline_stage: str = "retrieval",
        component_identifier: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            experiment_id=experiment_id,
            pipeline_stage=pipeline_stage,
            component_identifier=component_identifier,
            details=details,
        )


class ContextBudgetExceededError(RAGError):
    """Ngoại lệ khi kích thước ngữ cảnh vượt quá ngân sách token cho phép và chính sách là RAISE."""

    def __init__(
        self,
        message: str,
        experiment_id: Optional[str] = None,
        pipeline_stage: str = "context_assembly",
        component_identifier: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            experiment_id=experiment_id,
            pipeline_stage=pipeline_stage,
            component_identifier=component_identifier,
            details=details,
        )


class InvalidPromptError(RAGError):
    """Ngoại lệ khi dữ liệu đầu vào hoặc template prompt không hợp lệ."""

    def __init__(
        self,
        message: str,
        experiment_id: Optional[str] = None,
        pipeline_stage: str = "prompting",
        component_identifier: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            experiment_id=experiment_id,
            pipeline_stage=pipeline_stage,
            component_identifier=component_identifier,
            details=details,
        )


class LLMGenerationError(RAGError):
    """Ngoại lệ khi xảy ra lỗi trong quá trình suy luận hoặc gọi LLM Generator."""

    def __init__(
        self,
        message: str,
        experiment_id: Optional[str] = None,
        pipeline_stage: str = "generation",
        component_identifier: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            experiment_id=experiment_id,
            pipeline_stage=pipeline_stage,
            component_identifier=component_identifier,
            details=details,
        )


class MalformedStructuredOutputError(RAGError):
    """Ngoại lệ khi output của LLM không thể parse theo cấu trúc JSON/schema bắt buộc."""

    def __init__(
        self,
        message: str,
        experiment_id: Optional[str] = None,
        pipeline_stage: str = "output_parsing",
        component_identifier: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            experiment_id=experiment_id,
            pipeline_stage=pipeline_stage,
            component_identifier=component_identifier,
            details=details,
        )


class InvalidCitationError(RAGError):
    """Ngoại lệ khi trích dẫn pháp lý không hợp lệ về mặt cú pháp hoặc không tồn tại trong bằng chứng."""

    def __init__(
        self,
        message: str,
        experiment_id: Optional[str] = None,
        pipeline_stage: str = "citation_validation",
        component_identifier: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            experiment_id=experiment_id,
            pipeline_stage=pipeline_stage,
            component_identifier=component_identifier,
            details=details,
        )


class UnsupportedModelConfigurationError(RAGError):
    """Ngoại lệ khi cấu hình mô hình (ví dụ thinking_mode) không được hỗ trợ bởi model identifier."""

    def __init__(
        self,
        message: str,
        experiment_id: Optional[str] = None,
        pipeline_stage: str = "generation",
        component_identifier: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            experiment_id=experiment_id,
            pipeline_stage=pipeline_stage,
            component_identifier=component_identifier,
            details=details,
        )


class EvaluationDataLeakageError(RAGError):
    """Ngoại lệ nghiêm ngặt khi phát hiện vi phạm ranh giới frozen test (leakage detection)."""

    def __init__(
        self,
        message: str,
        experiment_id: Optional[str] = None,
        pipeline_stage: str = "dataset_protection",
        component_identifier: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            experiment_id=experiment_id,
            pipeline_stage=pipeline_stage,
            component_identifier=component_identifier,
            details=details,
        )


class IncompatibleComponentVersionError(RAGError):
    """Ngoại lệ khi phiên bản của các component không tương thích với nhau hoặc với schema."""

    def __init__(
        self,
        message: str,
        experiment_id: Optional[str] = None,
        pipeline_stage: Optional[str] = None,
        component_identifier: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            experiment_id=experiment_id,
            pipeline_stage=pipeline_stage,
            component_identifier=component_identifier,
            details=details,
        )


class ConfigurationValidationError(RAGError):
    """Ngoại lệ khi cấu hình thực nghiệm không hợp lệ hoặc vi phạm kiểm tra chéo (cross-validation)."""

    def __init__(
        self,
        message: str,
        experiment_id: Optional[str] = None,
        pipeline_stage: str = "configuration",
        component_identifier: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            experiment_id=experiment_id,
            pipeline_stage=pipeline_stage,
            component_identifier=component_identifier,
            details=details,
        )
