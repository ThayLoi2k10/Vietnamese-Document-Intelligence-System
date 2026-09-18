# scripts/benchmarks/build_chunking_dev_split.py
"""Tạo tập phân chia nội bộ Train/Dev từ retrieval_train.json cho Phase 7.

Tuân thủ nghiêm ngặt Mục 19.1 của Đặc tả Phase 7:
    - Tập 438 queries frozen test TUYỆT ĐỐI KHÔNG được sử dụng cho bất kỳ quyết định phát triển nào.
    - Tập 1,752 queries retrieval_train.json là nguồn phân tách duy nhất.
    - Phân chia phân tầng (stratified split trên single-label vs multi-label) theo tỷ lệ 80/20:
        * 1,401 queries: Train subset
        * 351 queries: Validation / Dev subset
    - Cố định seed ngẫu nhiên (seed = 42).
    - Xuất artifacts:
        * data/benchmarks/retrieval_dev_train.json
        * data/benchmarks/retrieval_dev_val.json
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List
from sklearn.model_selection import train_test_split

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.dense.artifacts import compute_file_sha256

TRAIN_SOURCE_PATH = PROJECT_ROOT / "data" / "benchmarks" / "retrieval_train.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "benchmarks"
DEV_TRAIN_PATH = OUTPUT_DIR / "retrieval_dev_train.json"
DEV_VAL_PATH = OUTPUT_DIR / "retrieval_dev_val.json"


def build_chunking_dev_split(
    source_path: Path = TRAIN_SOURCE_PATH,
    dev_train_path: Path = DEV_TRAIN_PATH,
    dev_val_path: Path = DEV_VAL_PATH,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> int:
    print("=" * 65)
    print("PHASE 7: TẠO PHÂN TẬP NỘI BỘ TRAIN / DEV (PREDECLARED SPLIT)")
    print("=" * 65)

    if not source_path.exists():
        print(f"[FAIL] Không tìm thấy train benchmark tại: {source_path}")
        return 1

    source_sha = compute_file_sha256(source_path)
    print(f"1. Nạp nguồn train: {source_path.name} (SHA-256: {source_sha[:16]}...)...")
    with open(source_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    total_queries = len(data)
    print(f"   -> Tổng số câu hỏi nguồn: {total_queries:,}")

    # Nhãn phân tầng: single-label (0) vs multi-label (1)
    stratify_labels = [1 if len(item["relevant_article_ids"]) > 1 else 0 for item in data]
    multi_count = sum(stratify_labels)
    print(f"   -> Phân bổ: Single-label={total_queries - multi_count:,}, Multi-label={multi_count:,} "
          f"({multi_count / total_queries:.2%})")

    # Phân chia phân tầng cố định
    train_subset, val_subset = train_test_split(
        data,
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify_labels,
    )

    print(f"2. Phân chia thành công (seed={seed}, val_ratio={val_ratio}):")
    print(f"   -> Dev Train : {len(train_subset):,} queries ({len(train_subset)/total_queries:.1%})")
    print(f"   -> Dev Val   : {len(val_subset):,} queries ({len(val_subset)/total_queries:.1%})")

    # Kiểm tra rò rỉ ID
    train_ids = {item["query_id"] for item in train_subset}
    val_ids = {item["query_id"] for item in val_subset}
    assert len(train_ids.intersection(val_ids)) == 0, "[FAIL] Rò rỉ query_id giữa train và val!"
    assert len(train_ids.union(val_ids)) == total_queries, "[FAIL] Số lượng query không bảo toàn!"

    # Lưu file
    with open(dev_train_path, "w", encoding="utf-8") as f:
        json.dump(train_subset, f, indent=2, ensure_ascii=False)
    print(f"   -> Đã ghi: {dev_train_path.name}")

    with open(dev_val_path, "w", encoding="utf-8") as f:
        json.dump(val_subset, f, indent=2, ensure_ascii=False)
    print(f"   -> Đã ghi: {dev_val_path.name}")

    print("=" * 65)
    print("[PASS] HOÀN TẤT TẠO PHÂN TẬP NỘI BỘ TRAIN / DEV THÀNH CÔNG")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(build_chunking_dev_split())
