"""Dense retriever BGE-M3 nhẹ cho serving: dùng `transformers` thuần thay vì FlagEmbedding.

Lý do tồn tại (bên cạnh `src/retrieval/dense/`):
- `FlagEmbedding` import `datasets` -> `pyarrow._dataset_parquet`. Trên một số máy Windows
  (Smart App Control / WDAC) DLL này bị chặn, khiến DenseRetriever gốc không khởi tạo được.
- BGE-M3 dense = CLS pooling + L2 normalize, nên có thể tái tạo chính xác bằng `transformers`
  (đã kiểm tra cosine ~1.0 với `embeddings.f32.npy` đã build ở Phase 6).

Chỉ phục vụ truy vấn (query encoding) trên embeddings đã build sẵn; không build lại corpus.
ID space giữ nguyên: hàng i của `embeddings.f32.npy` <-> `global_article_id = i`.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Union

import numpy as np
import pandas as pd

from src.retrieval.base import BaseRetriever, RetrievalResult

OFFICIAL_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024
EXPECTED_CORPUS_SIZE = 59636
DEFAULT_QUERY_MAX_LENGTH = 512


class HFDenseRetriever(BaseRetriever):
    def __init__(
        self,
        model_name_or_path: str = OFFICIAL_MODEL_NAME,
        device: Optional[str] = None,
        query_max_length: int = DEFAULT_QUERY_MAX_LENGTH,
    ) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.query_max_length = query_max_length
        self.index = None
        self.corpus_ids: List[int] = []
        self._tokenizer = None
        self._model = None

    def _init_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self._model = AutoModel.from_pretrained(self.model_name_or_path).to(self.device).eval()

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        import torch

        self._init_model()
        batch = self._tokenizer(
            [q or "" for q in queries],
            padding=True,
            truncation=True,
            max_length=self.query_max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            cls = self._model(**batch).last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, dim=-1)
        return cls.float().cpu().numpy().astype(np.float32)

    def fit(self, corpus: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Self:
        raise NotImplementedError(
            "HFDenseRetriever chỉ nạp embeddings đã build (Phase 6). Dùng load()."
        )

    def search(self, query_text: str, top_k: int = 10) -> List[RetrievalResult]:
        return self.batch_search([query_text], top_k=top_k)[0]

    def batch_search(self, queries: List[str], top_k: int = 10) -> List[List[RetrievalResult]]:
        self._check_is_fitted()
        if not queries:
            return []
        scores, idxs = self.index.search(self.encode_queries(queries), top_k)
        out: List[List[RetrievalResult]] = []
        for row_s, row_i in zip(scores, idxs):
            out.append(
                [
                    RetrievalResult(article_id=int(self.corpus_ids[i]), score=float(s), rank=r)
                    for r, (i, s) in enumerate(zip(row_i, row_s), start=1)
                    if i >= 0
                ]
            )
        return out

    def save(self, path: Union[str, Path]) -> None:
        raise NotImplementedError("HFDenseRetriever không ghi artifact; dùng pipeline Phase 6.")

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs: Any) -> Self:
        import json
        import faiss

        index_dir = Path(path)
        faiss_path = index_dir / "index.faiss"
        emb_path = index_dir / "embeddings.f32.npy"
        ids_path = index_dir / "corpus_ids.json"

        retriever = cls(**kwargs)
        if faiss_path.exists():
            retriever.index = faiss.read_index(str(faiss_path))
            if ids_path.exists():
                with open(ids_path, "r", encoding="utf-8") as f:
                    retriever.corpus_ids = [int(x) for x in json.load(f)]
            else:
                retriever.corpus_ids = list(range(retriever.index.ntotal))
        elif emb_path.exists():
            embeddings = np.load(emb_path)
            if embeddings.shape != (EXPECTED_CORPUS_SIZE, EMBEDDING_DIMENSION):
                raise ValueError(f"Shape embeddings không hợp lệ: {embeddings.shape}")
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
            retriever.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            retriever.index.add(embeddings)
            retriever.corpus_ids = list(range(embeddings.shape[0]))
        else:
            raise FileNotFoundError(
                f"Không tìm thấy index.faiss hoặc embeddings.f32.npy trong {index_dir}"
            )

        retriever.is_fitted = True
        return retriever
