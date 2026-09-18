# src/rag/reproducibility/manifest.py
"""Quản lý bản ghi tái lập thực nghiệm (ExperimentManifest) cho Phase 9 RAG.

Tuân thủ Mục 21: Reproducibility:
    - 21.1 Required metadata
    - 21.2 Component versioning
    - 21.3 Reproducibility manifest structure
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.rag.config.schemas.config_models import ExperimentConfig


@dataclass
class ExperimentManifest:
    """Bản khai sinh chi tiết lưu vết toàn bộ phiên bản và tham số của một thực nghiệm."""

    experiment_id: str
    created_at: str
    code_version: str
    dataset: Dict[str, Any]
    retriever: Dict[str, Any]
    reranker: Dict[str, Any]
    context: Dict[str, Any]
    prompt: Dict[str, Any]
    generation: Dict[str, Any]
    output_schema_version: str = "v1.0"
    citation_policy_version: str = "v1.0"
    abstention_policy_version: str = "v1.0"
    evaluation: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        """Ghi manifest ra tệp tin JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        output_schema_version: str = "v1.0",
        citation_policy_version: str = "v1.0",
        abstention_policy_version: str = "v1.0",
        notes: Optional[str] = None,
    ) -> "ExperimentManifest":
        """Khởi tạo ExperimentManifest từ cấu hình ExperimentConfig."""
        now_str = datetime.now(timezone.utc).isoformat()
        return cls(
            experiment_id=config.experiment_id,
            created_at=now_str,
            code_version=config.reproducibility.code_version,
            dataset=config.dataset.to_dict(),
            retriever=config.retriever.to_dict(),
            reranker=config.reranker.to_dict(),
            context=config.context.to_dict(),
            prompt=config.prompting.to_dict(),
            generation=config.generation.to_dict(),
            output_schema_version=output_schema_version,
            citation_policy_version=citation_policy_version,
            abstention_policy_version=abstention_policy_version,
            evaluation=config.evaluation.to_dict(),
            notes=notes or config.reproducibility.notes,
        )
