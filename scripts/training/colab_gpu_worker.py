"""GPU Training Worker chạy trên Google Colab (kết nối với máy Local qua Cloudflare Tunnel).

Chức năng:
1. Mở HTTP endpoints (`/health`, `/train_stream`, `/download_weights`).
2. Nhận dữ liệu `reranker_train_groups.jsonl` (1,401 nhóm Hard Negatives) và
   `reranker_val_candidates.jsonl` (351 câu Validation) gửi trực tiếp từ máy Local.
3. Gắn LoRA (`r=16, alpha=32`) vào toàn bộ các tầng Linear của `BAAI/bge-reranker-base`
   cùng tầng `classifier`, huấn luyện với hàm mất mát kết hợp:
   - ListNet / InfoNCE Cross-Entropy Loss trên nhóm (1 Positive vs K Hard Negatives)
   - Pairwise Margin Ranking Loss (ép điểm Positive > Hard Negative tối thiểu một khoảng margin)
4. Đánh giá trên 351 câu Validation sau mỗi epoch, tối ưu hệ số `fusion_alpha` và `pool_size`,
   stream log thời gian thực về Terminal máy Local, và trả file trọng số `lora_reranker_weights.pt`
   (~7 MB) về lưu tại `data/indices/reranker_finetuned/lora_reranker_weights.pt` trên máy Local.
"""

from __future__ import annotations

import gzip
import io
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, Generator, List, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

app = FastAPI(title="Vietnamese Legal Reranker Colab GPU Trainer")

BASE_MODEL_NAME = "BAAI/bge-reranker-base"
WEIGHTS_OUT_PATH = Path("/tmp/lora_reranker_weights.pt")


class LoRALinear(nn.Module):
    """Lớp bọc LoRA thuần PyTorch cho nn.Linear (không phụ thuộc phiên bản peft)."""

    def __init__(self, base_layer: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.05) -> None:
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / float(r)
        self.dropout = nn.Dropout(dropout)

        device = base_layer.weight.device
        self.lora_A = nn.Parameter(torch.empty((r, in_features), device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r), device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (self.dropout(x.float()) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return base_out + lora_out.to(base_out.dtype)


def inject_lora(model: nn.Module, r: int = 16, alpha: float = 32.0) -> List[str]:
    """Gắn LoRA vào các tầng attention & intermediate/output dense của RoBERTa."""
    target_suffixes = (
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense",
    )
    replaced: List[str] = []
    for p in model.parameters():
        p.requires_grad = False

    for name, module in list(model.named_modules()):
        if not name.startswith("roberta.encoder.layer."):
            continue
        if isinstance(module, nn.Linear) and name.endswith(target_suffixes):
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha))
            replaced.append(name)

    # Mở khóa tầng classifier head để học ranh giới quyết định cho văn bản luật
    for p in model.classifier.parameters():
        p.requires_grad = True

    return replaced


def extract_lora_and_classifier_state(
    model: nn.Module,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    lora_state: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A"] = module.lora_A.detach().cpu().half()
            lora_state[f"{name}.lora_B"] = module.lora_B.detach().cpu().half()

    cls_state: Dict[str, torch.Tensor] = {}
    for k, v in model.classifier.state_dict().items():
        # Trường hợp bên trong classifier có tầng nào được bọc hoặc giữ nguyên
        cls_state[k] = v.detach().cpu()
    return lora_state, cls_state


def compute_ndcg_at_10(retrieved_ids: List[int], relevant_set: set[int]) -> float:
    if not relevant_set:
        return 0.0
    dcg = 0.0
    for rank_idx, doc_id in enumerate(retrieved_ids[:10], start=1):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(rank_idx + 1)
    ideal_hits = min(len(relevant_set), 10)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr_at_10(retrieved_ids: List[int], relevant_set: set[int]) -> float:
    for rank_idx, doc_id in enumerate(retrieved_ids[:10], start=1):
        if doc_id in relevant_set:
            return 1.0 / rank_idx
    return 0.0


@torch.inference_mode()
def evaluate_validation(
    model: nn.Module,
    tokenizer: Any,
    val_records: List[Dict[str, Any]],
    device: str,
    max_length: int = 384,
) -> Dict[str, Any]:
    model.eval()
    all_query_ce_probs: List[List[float]] = []

    for rec in val_records:
        qtext = rec["query_text"]
        cands = rec["candidates"]
        texts = [c["text"] for c in cands]
        queries = [qtext] * len(texts)
        enc = tokenizer(
            queries,
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.cuda.amp.autocast(enabled=("cuda" in device)):
            logits = model(**enc).logits.squeeze(-1)
            if logits.ndim == 0:
                logits = logits.unsqueeze(0)
        probs = torch.sigmoid(logits.float()).cpu().tolist()
        all_query_ce_probs.append(probs)

    best_cfg = {"ndcg10": -1.0, "mrr10": 0.0, "hit1": 0.0, "hit3": 0.0, "alpha": 0.6, "pool_size": 10}

    for pool_size in (10, 15, 20):
        for alpha in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0):
            ndcg_sum = mrr_sum = hit1_sum = hit3_sum = 0.0
            for rec, ce_probs in zip(val_records, all_query_ce_probs):
                rel_set = set(rec["relevant_article_ids"])
                cands = rec["candidates"]
                pk = min(pool_size, len(cands))

                sub_ce = ce_probs[:pk]
                c_min, c_max = min(sub_ce), max(sub_ce)
                c_span = (c_max - c_min) if (c_max - c_min) > 1e-8 else 1.0
                ce_norm = [(x - c_min) / c_span for x in sub_ce]

                sub_pr = [float(c["hybrid_score"]) for c in cands[:pk]]
                p_min, p_max = min(sub_pr), max(sub_pr)
                p_span = (p_max - p_min) if (p_max - p_min) > 1e-8 else 1.0
                pr_norm = [(x - p_min) / p_span for x in sub_pr]

                scored = [
                    (cands[i]["article_id"], alpha * ce_norm[i] + (1.0 - alpha) * pr_norm[i])
                    for i in range(pk)
                ]
                scored.sort(key=lambda x: x[1], reverse=True)
                ranked_ids = [aid for aid, _ in scored] + [c["article_id"] for c in cands[pk:]]

                hit1_sum += float(bool(set(ranked_ids[:1]) & rel_set))
                hit3_sum += float(bool(set(ranked_ids[:3]) & rel_set))
                ndcg_sum += compute_ndcg_at_10(ranked_ids, rel_set)
                mrr_sum += compute_mrr_at_10(ranked_ids, rel_set)

            n = len(val_records)
            ndcg10 = ndcg_sum / n
            if ndcg10 > best_cfg["ndcg10"]:
                best_cfg = {
                    "ndcg10": round(ndcg10, 4),
                    "mrr10": round(mrr_sum / n, 4),
                    "hit1": round(hit1_sum / n * 100, 2),
                    "hit3": round(hit3_sum / n * 100, 2),
                    "alpha": alpha,
                    "pool_size": pool_size,
                }

    return best_cfg


@app.get("/health")
def health() -> JSONResponse:
    has_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        if has_cuda
        else 0.0
    )
    return JSONResponse(
        {
            "status": "ok",
            "cuda_available": has_cuda,
            "gpu_name": gpu_name,
            "vram_gb": vram_gb,
            "torch_version": torch.__version__,
        }
    )


@app.post("/train_stream")
async def train_stream(request: Request) -> StreamingResponse:
    raw_body = await request.body()
    payload = json.loads(gzip.decompress(raw_body).decode("utf-8"))

    train_groups: List[Dict[str, Any]] = payload["train_groups"]
    val_records: List[Dict[str, Any]] = payload["val_records"]
    cfg: Dict[str, Any] = payload.get("config", {})

    epochs = int(cfg.get("epochs", 3))
    lr = float(cfg.get("lr", 3e-4))
    lora_r = int(cfg.get("lora_r", 16))
    lora_alpha = float(cfg.get("lora_alpha", 32.0))
    max_length = int(cfg.get("max_length", 384))
    group_negs = int(cfg.get("group_negs", 4))
    grad_accum = int(cfg.get("grad_accum", 4))
    margin = float(cfg.get("margin", 1.0))

    def event_stream() -> Generator[bytes, None, None]:
        def emit(msg_type: str, **kwargs: Any) -> bytes:
            data = {"type": msg_type, "ts": round(time.time(), 2), **kwargs}
            return (json.dumps(data, ensure_ascii=False) + "\n").encode("utf-8")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        yield emit("info", message=f"Bắt đầu trên thiết bị: {gpu_name} ({device})")

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME)

        # Đo điểm Zero-shot Pretrained trên tập Val (351 câu) trước khi train
        model.to(device)
        zs_val = evaluate_validation(model, tokenizer, val_records, device, max_length=max_length)
        yield emit(
            "val_zero_shot",
            message=(
                f"[Trước khi Fine-tune - Zero-shot] Val nDCG@10={zs_val['ndcg10']} | "
                f"Hit@1={zs_val['hit1']}% | Hit@3={zs_val['hit3']}%"
            ),
            metrics=zs_val,
        )

        replaced_layers = inject_lora(model, r=lora_r, alpha=lora_alpha)
        model.to(device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        yield emit(
            "info",
            message=(
                f"Đã gắn LoRA (r={lora_r}, alpha={lora_alpha}) vào {len(replaced_layers)} tầng Linear + Classifier. "
                f"Tham số huấn luyện: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)"
            ),
        )

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=0.01,
        )
        total_steps = (len(train_groups) * epochs) // grad_accum
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(10, int(total_steps * 0.1)),
            num_training_steps=max(20, total_steps),
        )
        scaler = torch.cuda.amp.GradScaler(enabled=("cuda" in device))

        best_val = zs_val.copy()
        best_lora_state, best_cls_state = extract_lora_and_classifier_state(model)

        rng = random.Random(42)

        for epoch in range(1, epochs + 1):
            model.train()
            shuffled = list(train_groups)
            rng.shuffle(shuffled)
            running_loss = 0.0
            step_count = 0
            optimizer.zero_grad()
            t_ep = time.time()

            for idx, grp in enumerate(shuffled, start=1):
                qtext = grp["query_text"]
                pos_item = rng.choice(grp["positives"])
                negs = grp["hard_negatives"]
                if len(negs) > group_negs:
                    # Ưu tiên lấy các hard negative ở thứ hạng cao nhất + lấy mẫu ngẫu nhiên
                    sampled_negs = negs[:2] + rng.sample(negs[2:], group_negs - 2)
                else:
                    sampled_negs = list(negs)

                texts = [pos_item["text"]] + [n["text"] for n in sampled_negs]
                queries = [qtext] * len(texts)

                enc = tokenizer(
                    queries,
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)

                with torch.cuda.amp.autocast(enabled=("cuda" in device)):
                    logits = model(**enc).logits.squeeze(-1)
                    # 1. ListNet Cross-Entropy: ép xác suất của ứng viên Positive (index 0) tiến tới 1
                    target = torch.zeros(1, dtype=torch.long, device=device)
                    ce_loss = F.cross_entropy(logits.unsqueeze(0), target)
                    # 2. Pairwise Margin Loss: ép logit(pos) - logit(neg_i) >= margin
                    pos_logit = logits[0]
                    neg_logits = logits[1:]
                    margin_loss = F.relu(margin - (pos_logit - neg_logits)).mean()
                    loss = (ce_loss + 0.5 * margin_loss) / grad_accum

                scaler.scale(loss).backward()
                running_loss += float(loss.item()) * grad_accum
                step_count += 1

                if idx % grad_accum == 0 or idx == len(shuffled):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                if idx % 200 == 0 or idx == len(shuffled):
                    avg_l = running_loss / max(1, step_count)
                    yield emit(
                        "progress",
                        epoch=epoch,
                        step=idx,
                        total=len(shuffled),
                        loss=round(avg_l, 4),
                        elapsed_sec=round(time.time() - t_ep, 1),
                    )

            val_metrics = evaluate_validation(
                model, tokenizer, val_records, device, max_length=max_length
            )
            improved = val_metrics["ndcg10"] >= best_val["ndcg10"]
            if improved:
                best_val = val_metrics
                best_lora_state, best_cls_state = extract_lora_and_classifier_state(model)

            yield emit(
                "epoch_done",
                epoch=epoch,
                val_metrics=val_metrics,
                is_best=improved,
                message=(
                    f"Epoch {epoch}/{epochs} hoàn tất | Val nDCG@10={val_metrics['ndcg10']} "
                    f"| MRR@10={val_metrics['mrr10']} | Hit@1={val_metrics['hit1']}% "
                    f"| Hit@3={val_metrics['hit3']}% (alpha={val_metrics['alpha']}, pool={val_metrics['pool_size']})"
                    + (" [NEW BEST ✅]" if improved else "")
                ),
            )

        ckpt_obj = {
            "base_model_name": BASE_MODEL_NAME,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "best_fusion_alpha": best_val["alpha"],
            "best_pool_size": best_val["pool_size"],
            "lora_state_dict": best_lora_state,
            "classifier_state_dict": best_cls_state,
            "metadata": {
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "gpu_name": gpu_name,
                "epochs": epochs,
                "lr": lr,
                "train_queries": len(train_groups),
                "val_queries": len(val_records),
                "val_zero_shot": zs_val,
                "val_best_finetuned": best_val,
            },
        }
        torch.save(ckpt_obj, WEIGHTS_OUT_PATH)
        size_mb = round(WEIGHTS_OUT_PATH.stat().st_size / (1024 * 1024), 2)
        yield emit(
            "completed",
            weights_size_mb=size_mb,
            best_val=best_val,
            zero_shot_val=zs_val,
            message=f"Huấn luyện hoàn tất! Trọng số đã sẵn sàng tải về ({size_mb} MB).",
        )

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.get("/download_weights")
def download_weights() -> FileResponse:
    if not WEIGHTS_OUT_PATH.exists():
        raise RuntimeError("Chưa có file trọng số.")
    return FileResponse(
        path=str(WEIGHTS_OUT_PATH),
        filename="lora_reranker_weights.pt",
        media_type="application/octet-stream",
    )
