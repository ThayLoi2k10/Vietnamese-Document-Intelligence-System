"""Mô hình truy xuất thông tin Okapi BM25 sử dụng inverted index."""

import math
from pathlib import Path
import pickle
import re
from typing import Any, Dict, List, Self, Union
import numpy as np
import pandas as pd

from src.retrieval.base import BaseRetriever, RetrievalResult, extract_corpus_columns


def default_tokenizer(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


class BM25Retriever(BaseRetriever):
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        super().__init__()
        if k1 < 0:
            raise ValueError("Tham số k1 phải >= 0.")
        if not (0.0 <= b <= 1.0):
            raise ValueError("Tham số b phải nằm trong đoạn [0, 1].")

        self.k1 = float(k1)
        self.b = float(b)
        self.epsilon = float(epsilon)

        self.article_ids: List[int] = []
        self.doc_lengths: np.ndarray = np.array([], dtype=np.float32)
        self.avgdl: float = 0.0
        self.corpus_size: int = 0
        self.inverted_index: Dict[str, Dict[str, np.ndarray]] = {}
        self.idf: Dict[str, float] = {}

    def fit(self, corpus: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Self:
        self.article_ids, raw_texts = extract_corpus_columns(corpus)
        self.corpus_size = len(self.article_ids)

        doc_lens = []
        term_doc_freqs: Dict[str, List[int]] = {}
        term_tfs: Dict[str, List[int]] = {}

        for doc_idx, text in enumerate(raw_texts):
            tokens = default_tokenizer(text)
            doc_lens.append(len(tokens))

            tf_counter: Dict[str, int] = {}
            for t in tokens:
                tf_counter[t] = tf_counter.get(t, 0) + 1

            for term, count in tf_counter.items():
                if term not in term_doc_freqs:
                    term_doc_freqs[term] = []
                    term_tfs[term] = []
                term_doc_freqs[term].append(doc_idx)
                term_tfs[term].append(count)

        self.doc_lengths = np.array(doc_lens, dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_lengths)) if self.corpus_size > 0 else 0.0

        self.inverted_index = {}
        self.idf = {}

        for term, doc_list in term_doc_freqs.items():
            df_val = len(doc_list)
            idf_val = math.log(
                1.0 + (self.corpus_size - df_val + 0.5) / (df_val + 0.5)
            )
            self.idf[term] = idf_val

            self.inverted_index[term] = {
                "docs": np.array(doc_list, dtype=np.int32),
                "tfs": np.array(term_tfs[term], dtype=np.float32),
            }

        self.is_fitted = True
        return self

    def search(self, query_text: str, top_k: int = 10) -> List[RetrievalResult]:
        self._check_is_fitted()

        if top_k <= 0:
            raise ValueError("top_k phải lớn hơn 0.")

        if self.corpus_size == 0 or self.avgdl <= 0:
            return []

        tokens = default_tokenizer(query_text)
        if not tokens:
            return []

        scores = np.zeros(self.corpus_size, dtype=np.float32)

        for term in set(tokens):
            if term not in self.inverted_index:
                continue

            posting = self.inverted_index[term]
            doc_idxs = posting["docs"]
            tfs = posting["tfs"]
            idf_val = self.idf[term]

            lens = self.doc_lengths[doc_idxs]
            denom = tfs + self.k1 * (1.0 - self.b + self.b * (lens / self.avgdl))
            term_scores = idf_val * ((tfs * (self.k1 + 1.0)) / denom)

            scores[doc_idxs] += term_scores

        actual_k = min(top_k, self.corpus_size)
        top_k_indices = np.argpartition(scores, -actual_k)[-actual_k:]
        sorted_indices = top_k_indices[np.argsort(-scores[top_k_indices])]

        results: List[RetrievalResult] = []
        for rank, idx in enumerate(sorted_indices, start=1):
            score_val = float(scores[idx])
            if score_val <= 0.0:
                break
            results.append(
                RetrievalResult(
                    article_id=self.article_ids[idx],
                    score=score_val,
                    rank=rank,
                )
            )

        return results

    def save(self, path: Union[str, Path]) -> None:
        self._check_is_fitted()
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
            "article_ids": self.article_ids,
            "doc_lengths": self.doc_lengths,
            "avgdl": self.avgdl,
            "corpus_size": self.corpus_size,
            "inverted_index": self.inverted_index,
            "idf": self.idf,
        }
        temp_path = save_path.with_suffix(save_path.suffix + ".tmp")
        try:
            with open(temp_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path.replace(save_path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Union[str, Path]) -> Self:
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Không tìm thấy index tại: {load_path}")
        with open(load_path, "rb") as f:
            payload = pickle.load(f)

        instance = cls(
            k1=payload["k1"],
            b=payload["b"],
            epsilon=payload.get("epsilon", 0.25),
        )
        instance.article_ids = payload["article_ids"]
        instance.doc_lengths = payload["doc_lengths"]
        instance.avgdl = payload["avgdl"]
        instance.corpus_size = payload["corpus_size"]
        instance.inverted_index = payload["inverted_index"]
        instance.idf = payload["idf"]
        instance.is_fitted = True
        return instance