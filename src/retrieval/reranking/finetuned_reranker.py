"""Bộ tái xếp hạng Cross-Encoder đã Fine-tune (LoRA + Residual Hybrid Fusion) dùng thuần `transformers`.

Ưu điểm kiến trúc:
1. Nạp trực tiếp `BAAI/bge-reranker-base` qua `transformers` (`AutoModelForSequenceClassification`),
   không import `FlagEmbedding` hay `datasets`, tránh hoàn toàn lỗi chặn DLL của Windows Application Control.
2. Áp dụng trực tiếp trọng số LoRA (`lora_A`, `lora_B`) và `classifier` head đã huấn luyện trên
   GPU Colab từ tệp `data/indices/reranker_finetuned/lora_reranker_weights.pt` vào mô hình gốc bằng PyTorch thuần.
3. Kết hợp điểm Cross-Encoder đã fine-tune với điểm ưu tiên Tầng 1 (Hybrid RRF Prior) theo hệ số
   `fusion_alpha` được tối ưu trên tập Validation (`retrieval_dev_val.json`), chống hiện tượng đảo ngược
   thứ hạng (Pool Depth Inversion) đã ghi nhận ở Phase 12.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence
import math

import torch

from src.retrieval.base import RetrievalResult

BASE_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_FINETUNED_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "indices"
    / "reranker_finetuned"
    / "lora_reranker_weights.pt"
)


class FinetunedLegalReranker:
    """Cross-Encoder Reranker đã fine-tune chuyên biệt cho Pháp luật Việt Nam."""

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        base_model_name: str = BASE_RERANKER_MODEL,
        device: Optional[str] = None,
        max_length: int = 384,
        batch_size: int = 16,
    ) -> None:
        self.weights_path = Path(weights_path or DEFAULT_FINETUNED_WEIGHTS_PATH)
        self.base_model_name = base_model_name
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size

        self._tokenizer = None
        self._model = None
        self.is_finetuned: bool = False
        self.fusion_alpha: float = 0.65
        self.candidate_pool_size: int = 10
        self.training_metadata: Dict[str, object] = {}

    def _init_model(self) -> None:
        if self._model is not None:
            return

        import gc
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            low_cpu_mem_usage=True,
        )

        if self.weights_path.exists():
            ckpt = torch.load(self.weights_path, map_location="cpu", weights_only=False)
            lora_state: Dict[str, torch.Tensor] = ckpt.get("lora_state_dict", {})
            classifier_state: Dict[str, torch.Tensor] = ckpt.get("classifier_state_dict", {})
            lora_r = float(ckpt.get("lora_r", 16))
            lora_alpha = float(ckpt.get("lora_alpha", 32.0))
            scaling = lora_alpha / lora_r

            self.fusion_alpha = float(ckpt.get("best_fusion_alpha", 0.65))
            self.candidate_pool_size = int(ckpt.get("best_pool_size", 10))
            self.training_metadata = ckpt.get("metadata", {})

            # Gộp (merge) trực tiếp ma trận LoRA ΔW = (B @ A) * scaling vào các tầng Linear của RoBERTa
            modules_dict = dict(model.named_modules())
            merged_layers = 0
            with torch.no_grad():
                for key, lora_A in lora_state.items():
                    if not key.endswith(".lora_A"):
                        continue
                    module_name = key[: -len(".lora_A")]
                    lora_B = lora_state.get(f"{module_name}.lora_B")
                    if lora_B is None or module_name not in modules_dict:
                        continue
                    target_mod = modules_dict[module_name]
                    if hasattr(target_mod, "weight"):
                        delta_w = (lora_B.float() @ lora_A.float()) * scaling
                        target_mod.weight.data.add_(delta_w.to(target_mod.weight.dtype))
                        merged_layers += 1

                # Cập nhật tầng phân loại (classifier head) đã fine-tune
                if classifier_state:
                    model.classifier.load_state_dict(classifier_state, strict=False)

            self.is_finetuned = merged_layers > 0
            del ckpt, lora_state, classifier_state

        target_dtype = torch.float16 if "cuda" in str(self.device) else torch.bfloat16
        self._model = model.to(device=self.device, dtype=target_dtype).eval()
        gc.collect()

    def score_pairs(self, query: str, texts: Sequence[str]) -> List[float]:
        """Tính điểm logit cho danh sách (query, điều luật) — sắp xếp theo độ dài để giảm padding trên CPU."""
        if not texts:
            return []
        self._init_model()
        indexed = sorted(enumerate(texts), key=lambda x: len(x[1]))
        scores_out = [0.0] * len(texts)
        for start in range(0, len(indexed), self.batch_size):
            chunk = indexed[start : start + self.batch_size]
            batch_texts = [t for _, t in chunk]
            batch_queries = [query] * len(chunk)
            encoded = self._tokenizer(
                batch_queries,
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self._model(**encoded).logits.squeeze(-1)
                if logits.ndim == 0:
                    logits = logits.unsqueeze(0)
            for (orig_idx, _), val in zip(chunk, logits.float().cpu().tolist()):
                scores_out[orig_idx] = float(val)
        return scores_out

    def rerank(
        self,
        query: str,
        candidate_ids: Sequence[int],
        candidate_texts: Sequence[str],
        candidate_prior_scores: Optional[Sequence[float]] = None,
        candidate_pool_size: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        if len(candidate_ids) != len(candidate_texts):
            raise ValueError("Số lượng candidate_ids và candidate_texts không khớp.")
        if not candidate_ids:
            return []

        self._init_model()
        pool_k = min(
            candidate_pool_size or self.candidate_pool_size,
            len(candidate_ids),
        )
        pool_ids = list(candidate_ids[:pool_k])
        pool_texts = list(candidate_texts[:pool_k])

        raw_logits = self.score_pairs(query, pool_texts)
        ce_probs = [1.0 / (1.0 + math.exp(-max(min(x, 20.0), -20.0))) for x in raw_logits]

        # Chuẩn hóa min-max điểm Cross-Encoder trong pool
        ce_min, ce_max = min(ce_probs), max(ce_probs)
        ce_span = (ce_max - ce_min) if (ce_max - ce_min) > 1e-8 else 1.0
        ce_norm = [(p - ce_min) / ce_span for p in ce_probs]

        # Chuẩn hóa điểm ưu tiên Tầng 1 (Hybrid RRF Prior)
        if candidate_prior_scores is not None and len(candidate_prior_scores) >= pool_k:
            priors = [float(s) for s in candidate_prior_scores[:pool_k]]
        else:
            priors = [1.0 / (5.0 + r) for r in range(1, pool_k + 1)]
        p_min, p_max = min(priors), max(priors)
        p_span = (p_max - p_min) if (p_max - p_min) > 1e-8 else 1.0
        prior_norm = [(p - p_min) / p_span for p in priors]

        alpha = self.fusion_alpha
        combined = [
            (aid, alpha * c_sc + (1.0 - alpha) * p_sc)
            for aid, c_sc, p_sc in zip(pool_ids, ce_norm, prior_norm)
        ]
        combined.sort(key=lambda x: x[1], reverse=True)

        results: List[RetrievalResult] = [
            RetrievalResult(article_id=int(aid), score=float(sc), rank=idx)
            for idx, (aid, sc) in enumerate(combined, start=1)
        ]

        # Giữ nguyên thứ tự các ứng viên ngoài pool_k
        min_score = min((r.score for r in results), default=0.0)
        for offset, rem_id in enumerate(candidate_ids[pool_k:], start=1):
            results.append(
                RetrievalResult(
                    article_id=int(rem_id),
                    score=float(min_score - offset * 0.001),
                    rank=pool_k + offset,
                )
            )

        if top_k is not None:
            results = results[:top_k]
        return results
