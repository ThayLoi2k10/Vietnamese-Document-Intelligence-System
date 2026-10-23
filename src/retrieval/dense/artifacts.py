"""Quản lý lưu trữ, fingerprint và tính toàn vẹn artifact cho mô hình dense retrieval."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

EXPECTED_CORPUS_SIZE = 59636
EXPECTED_DIMENSION = 1024


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_corpus_fingerprint(records: List[Tuple[int, str]]) -> str:
    hasher = hashlib.sha256()
    for article_id, text in records:
        hasher.update(f"{article_id}:{text}\n".encode("utf-8"))
    return hasher.hexdigest()


def compute_id_sequence_fingerprint(ids: List[int]) -> str:
    hasher = hashlib.sha256()
    for cid in ids:
        hasher.update(f"{cid}\n".encode("utf-8"))
    return hasher.hexdigest()


class DenseArtifactManager:
    def __init__(self, artifact_dir: Union[str, Path]) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.config_path = self.artifact_dir / "config.json"
        self.corpus_ids_path = self.artifact_dir / "corpus_ids.json"
        self.corpus_fingerprint_path = self.artifact_dir / "corpus_fingerprint.json"
        self.embeddings_path = self.artifact_dir / "embeddings.f32.npy"
        self.index_path = self.artifact_dir / "index.faiss"
        self.build_report_path = self.artifact_dir / "build_report.json"

    def _atomic_write_json(self, file_path: Path, data: Any) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=file_path.parent, delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(data, tmp, indent=2, ensure_ascii=False)
            tmp_name = tmp.name
        os.replace(tmp_name, file_path)

    def _atomic_write_numpy(self, file_path: Path, array: np.ndarray) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "wb", dir=file_path.parent, delete=False
        ) as tmp:
            np.save(tmp, array)
            tmp_name = tmp.name
        os.replace(tmp_name, file_path)

    def save_all_artifacts(
        self,
        config: Dict[str, Any],
        corpus_ids: List[int],
        corpus_fingerprint: Dict[str, Any],
        embeddings: np.ndarray,
        faiss_index_saver: Any,
        build_report: Dict[str, Any],
    ) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write_json(self.config_path, config)
        self._atomic_write_json(self.corpus_ids_path, corpus_ids)
        self._atomic_write_json(self.corpus_fingerprint_path, corpus_fingerprint)
        self._atomic_write_numpy(self.embeddings_path, embeddings)

        temp_index = self.artifact_dir / "temp_index.faiss"
        faiss_index_saver(temp_index)
        os.replace(temp_index, self.index_path)

        self._atomic_write_json(self.build_report_path, build_report)

    def verify_artifacts(
        self,
        expected_corpus_fingerprint: Optional[str] = None,
        expected_count: int = EXPECTED_CORPUS_SIZE,
    ) -> Dict[str, Any]:
        required_files = [
            self.config_path,
            self.corpus_ids_path,
            self.corpus_fingerprint_path,
            self.embeddings_path,
            self.index_path,
        ]

        for path in required_files:
            if not path.exists():
                raise FileNotFoundError(f"[FAIL FAST] Thiếu artifact bắt buộc: {path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        with open(self.corpus_ids_path, "r", encoding="utf-8") as f:
            corpus_ids = json.load(f)

        if len(corpus_ids) != expected_count:
            raise ValueError(
                f"[FAIL FAST] Số lượng corpus_ids ({len(corpus_ids)}) != {expected_count}"
            )

        with open(self.corpus_fingerprint_path, "r", encoding="utf-8") as f:
            fp_data = json.load(f)

        if expected_corpus_fingerprint is not None:
            actual_fp = fp_data.get("corpus_text_id_sha256")
            if actual_fp != expected_corpus_fingerprint:
                raise ValueError(
                    f"[FAIL FAST] Sai lệch corpus fingerprint!\n"
                    f"Kỳ vọng: {expected_corpus_fingerprint}\n"
                    f"Thực tế: {actual_fp}"
                )

        mmap_arr = np.load(self.embeddings_path, mmap_mode="r")
        if mmap_arr.shape != (expected_count, EXPECTED_DIMENSION):
            raise ValueError(
                f"[FAIL FAST] Kích thước embeddings {mmap_arr.shape} != "
                f"({expected_count}, {EXPECTED_DIMENSION})"
            )
        if mmap_arr.dtype != np.float32:
            raise ValueError(
                f"[FAIL FAST] Kiểu dữ liệu embeddings {mmap_arr.dtype} != float32"
            )

        return {
            "status": "PASS",
            "config": config,
            "corpus_ids_count": len(corpus_ids),
            "fingerprint": fp_data,
            "embeddings_shape": list(mmap_arr.shape),
            "embeddings_dtype": str(mmap_arr.dtype),
        }
