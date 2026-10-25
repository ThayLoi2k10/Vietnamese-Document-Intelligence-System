r"""Script điều khiển huấn luyện Cross-Encoder Reranker từ máy Local qua Cloudflare Tunnel lên GPU Colab.

Cách chạy trên máy Local (PowerShell):
    .\.venv\Scripts\python.exe scripts\training\train_reranker_tunnel.py --colab-url https://xxxx.trycloudflare.com

Quy trình tự động:
1. Kiểm tra kết nối `/health` tới GPU Worker trên Colab (xác nhận GPU T4/L4/A100).
2. Đọc `data/training/reranker_train_groups.jsonl` (1,401 nhóm Hard Negatives) và
   `data/training/reranker_val_candidates.jsonl` (351 câu Validation) tại máy Local,
   nén gzip (~4.5 MB) và đẩy qua HTTPS Tunnel lên GPU Colab.
3. Hiển thị trực tiếp tiến trình huấn luyện (Loss từng bước, nDCG@10, MRR@10, Hit@1%, Hit@3%
   trên tập Validation sau mỗi Epoch) ngay trên cửa sổ Terminal của máy Local.
4. Khi huấn luyện xong, tự động tải trọng số `lora_reranker_weights.pt` (~7 MB) về lưu tại
   `data/indices/reranker_finetuned/lora_reranker_weights.pt` và lưu báo cáo huấn luyện tại
   `results/benchmarks/reranker_training_report.json`.
"""

import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import requests

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def read_env_colab_url() -> str:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return ""
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if k.strip() == "COLAB_URL":
                return v.strip().strip("'\"")
    return ""


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Huấn luyện LoRA Cross-Encoder Reranker từ máy Local qua Colab GPU Tunnel."
    )
    parser.add_argument(
        "--colab-url",
        type=str,
        default="",
        help="URL Cloudflare Tunnel từ Colab (ví dụ: https://xxxx.trycloudflare.com)",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Số epoch huấn luyện (mặc định: 3)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate cho LoRA (mặc định: 2e-4)")
    parser.add_argument("--lora-r", type=int, default=16, help="Bậc LoRA r (mặc định: 16)")
    parser.add_argument("--lora-alpha", type=float, default=32.0, help="Hệ số LoRA alpha (mặc định: 32)")
    parser.add_argument("--max-length", type=int, default=384, help="Độ dài token tối đa (mặc định: 384)")
    args = parser.parse_args()

    base_url = (args.colab_url or os.environ.get("COLAB_URL") or read_env_colab_url()).strip().rstrip("/")
    if not base_url:
        print("❌ Lỗi: Chưa cung cấp URL Colab Tunnel.")
        print("Hãy chạy: .\\.venv\\Scripts\\python.exe scripts\\training\\train_reranker_tunnel.py --colab-url https://xxxx.trycloudflare.com")
        sys.exit(1)

    train_path = PROJECT_ROOT / "data" / "training" / "reranker_train_groups.jsonl"
    val_path = PROJECT_ROOT / "data" / "training" / "reranker_val_candidates.jsonl"
    if not train_path.exists() or not val_path.exists():
        print("❌ Chưa có dữ liệu huấn luyện tại data/training/. Đang chạy build_reranker_dataset.py...")
        from scripts.training.build_reranker_dataset import main as build_ds

        build_ds()

    print("=" * 80)
    print("BƯỚC 2: HUẤN LUYỆN LORA CROSS-ENCODER RERANKER QUA COLAB GPU TUNNEL")
    print("=" * 80)
    print(f"🔗 Đang kiểm tra kết nối tới GPU Worker: {base_url}/health ...")

    try:
        h_resp = requests.get(f"{base_url}/health", timeout=20)
        h_resp.raise_for_status()
        h_info = h_resp.json()
    except Exception as exc:
        print(f"❌ Không kết nối được tới Colab Tunnel ({base_url}): {exc}")
        print("Hãy kiểm tra lại xem ô code trên Google Colab đã báo link *.trycloudflare.com chưa.")
        sys.exit(1)

    print(
        f"✅ Kết nối thành công! Thiết bị Colab: {h_info.get('gpu_name')} "
        f"(VRAM: {h_info.get('vram_gb')} GB | CUDA: {h_info.get('cuda_available')} | PyTorch: {h_info.get('torch_version')})"
    )
    if not h_info.get("cuda_available"):
        print("⚠️ Cảnh báo: Colab hiện đang ở chế độ CPU! Hãy đổi Runtime trên Colab sang T4 GPU trước khi train.")
        sys.exit(1)

    train_groups = load_jsonl(train_path)
    val_records = load_jsonl(val_path)
    payload = {
        "train_groups": train_groups,
        "val_records": val_records,
        "config": {
            "epochs": args.epochs,
            "lr": args.lr,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "max_length": args.max_length,
            "group_negs": 4,
            "grad_accum": 4,
            "margin": 1.0,
        },
    }
    compressed = gzip.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    print(
        f"📦 Đang gửi {len(train_groups)} nhóm Train + {len(val_records)} câu Val "
        f"(đã nén: {len(compressed) / (1024 * 1024):.2f} MB) lên GPU Colab..."
    )

    final_summary: Dict[str, Any] = {}
    epoch_logs: List[Dict[str, Any]] = []

    with requests.post(
        f"{base_url}/train_stream",
        data=compressed,
        headers={"Content-Type": "application/octet-stream"},
        stream=True,
        timeout=(60, 1800),
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            event = json.loads(raw_line.decode("utf-8", errors="replace"))
            etype = event.get("type")
            if etype in ("info", "val_zero_shot"):
                print(f"ℹ️  {event.get('message')}")
            elif etype == "progress":
                print(
                    f"   [Epoch {event['epoch']}/{args.epochs}] Bước {event['step']:4d}/{event['total']} "
                    f"| Loss: {event['loss']:.4f} | Thời gian: {event['elapsed_sec']}s"
                )
            elif etype == "epoch_done":
                print(f"🎯 {event.get('message')}")
                epoch_logs.append(event)
            elif etype == "completed":
                print(f"✅ {event.get('message')}")
                final_summary = event

    out_weights_dir = PROJECT_ROOT / "data" / "indices" / "reranker_finetuned"
    out_weights_dir.mkdir(parents=True, exist_ok=True)
    out_weights_path = out_weights_dir / "lora_reranker_weights.pt"

    print(f"⬇️  Đang tải trọng số mô hình đã huấn luyện về {out_weights_path} ...")
    with requests.get(f"{base_url}/download_weights", stream=True, timeout=120) as d_resp:
        d_resp.raise_for_status()
        with open(out_weights_path, "wb") as f:
            for chunk in d_resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

    report_path = PROJECT_ROOT / "results" / "benchmarks" / "reranker_training_report.json"
    report_data = {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "colab_gpu": h_info,
        "hyperparameters": payload["config"],
        "epoch_history": epoch_logs,
        "summary": final_summary,
        "saved_weights_path": str(out_weights_path.relative_to(PROJECT_ROOT)),
        "saved_weights_bytes": out_weights_path.stat().st_size,
    }
    report_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 80)
    print(f"🎉 ĐÃ LƯU TRỌNG SỐ MODEL TẠI: {out_weights_path} ({out_weights_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"📊 Báo cáo quá trình huấn luyện: {report_path}")
    print("👉 Bước tiếp theo: Chạy đánh giá trên 438 câu Frozen Test & chấm % trả lời câu hỏi ở Local:")
    print("   .\\.venv\\Scripts\\python.exe scripts\\evaluation\\evaluate_finetuned_rag.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
