# scripts/benchmarks/build_retrieval_benchmark.py
"""Script xây dựng Retrieval Benchmark (Train / Test Split).

Thực hiện:
    - Đọc tập query gốc từ train.json của legal_retrieval
    - Kiểm tra tính hợp lệ của qid (không None, không trùng lặp)
    - Chuẩn hóa text query (Unicode NFC, chuẩn hóa whitespace)
    - Fail fast nếu phát hiện query rỗng hoặc không có relevant article (ground truth)
    - Phân tầng Stratified Split theo độ phức tạp (single-label vs multi-label)
    - Kiểm định tính phân hoạch: Train ∩ Test = ∅ và Train ∪ Test = All
    - Ghi artifact thông qua temporary files và atomic replace

Input:  data/raw/legal_retrieval/train.json
Output: data/benchmarks/retrieval_train.json
        data/benchmarks/retrieval_test.json
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import unicodedata
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
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "legal_retrieval" / "train.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "benchmarks"
TRAIN_OUTPUT = OUTPUT_DIR / "retrieval_train.json"
TEST_OUTPUT = OUTPUT_DIR / "retrieval_test.json"


def clean_text(text: Any) -> str:
    """Chuẩn hóa Unicode NFC và gộp khoảng trắng thừa."""
    if not isinstance(text, str) or not text.strip():
        return ""
    normalized = unicodedata.normalize("NFC", text)
    return " ".join(normalized.split())


def build_retrieval_benchmark(
    force: bool = False,
    test_size: float = 0.20,
    seed: int = 42,
) -> None:
    """Xây dựng và chia tập benchmark cho bài toán retrieval."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file nguồn: {INPUT_PATH}. "
            "Hãy đảm bảo raw dataset legal_retrieval đã tồn tại."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Kiểm tra artifact đã tồn tại
    if (
        not force
        and TRAIN_OUTPUT.exists()
        and TRAIN_OUTPUT.stat().st_size > 0
        and TEST_OUTPUT.exists()
        and TEST_OUTPUT.stat().st_size > 0
    ):
        print(
            f"Artifacts benchmark đã tồn tại:\n"
            f"  - {TRAIN_OUTPUT}\n"
            f"  - {TEST_OUTPUT}\n"
            "Bỏ qua xử lý (Dùng --force để tạo lại)."
        )
        return

    print("=" * 60)
    print("XÂY DỰNG RETRIEVAL BENCHMARK (TRAIN / TEST SPLIT)")
    print("=" * 60)

    print(f"Đọc dữ liệu thô từ: {INPUT_PATH}...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw_queries = json.load(f)

    if not raw_queries:
        raise ValueError("Tập dữ liệu truy vấn thô rỗng, không thể tạo benchmark.")

    print(f"Tổng số query thô nạp vào: {len(raw_queries):,}")

    # 1. Validation sơ bộ qid trên toàn bộ raw queries
    query_ids = [item.get("qid") for item in raw_queries]
    if any(qid is None for qid in query_ids):
        raise ValueError("Phát hiện query không có qid.")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Phát hiện qid bị trùng lặp.")

    processed_queries = []
    stratify_labels = []

    # 2. Tiền xử lý và fail fast nếu vi phạm tính toàn vẹn ground truth
    for item in raw_queries:
        qid = item.get("qid")
        raw_text = item.get("question", "")
        raw_labels = item.get("relevant_laws", [])

        clean_qtext = clean_text(raw_text)
        if not clean_qtext:
            raise ValueError(f"Query rỗng sau preprocessing: qid={qid}")

        clean_labels = sorted(list(set(raw_labels)))
        if not clean_labels:
            raise ValueError(f"Query không có relevant article: qid={qid}")

        # Phân tầng: 0 nếu single-label (1), 1 nếu multi-label (>1)
        is_multi = 1 if len(clean_labels) > 1 else 0
        stratify_labels.append(is_multi)

        processed_queries.append(
            {
                "query_id": qid,
                "query_text": clean_qtext,
                "relevant_article_ids": clean_labels,
            }
        )

    # 3. Stratified Split 80/20
    train_queries, test_queries = train_test_split(
        processed_queries,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_labels,
    )

    print("\n" + "-" * 20 + " THỐNG KÊ PHÂN TẬP " + "-" * 20)
    total_q = len(processed_queries)
    print(f"Tập Train: {len(train_queries):,} queries ({len(train_queries)/total_q*100:.1f}%)")
    print(f"Tập Test : {len(test_queries):,} queries ({len(test_queries)/total_q*100:.1f}%)")

    train_multi = sum(1 for q in train_queries if len(q["relevant_article_ids"]) > 1)
    test_multi = sum(1 for q in test_queries if len(q["relevant_article_ids"]) > 1)

    print(f"\nPhân phối Multi-label (>1 article liên quan):")
    print(f"  - Train: {train_multi:,} / {len(train_queries):,} ({train_multi/len(train_queries)*100:.2f}%)")
    print(f"  - Test : {test_multi:,} / {len(test_queries):,} ({test_multi/len(test_queries)*100:.2f}%)")

    # 4. Kiểm tra phân hoạch chặt chẽ: Train ∩ Test = ∅ và Train ∪ Test = All
    all_ids = set(q["query_id"] for q in processed_queries)
    train_ids = set(q["query_id"] for q in train_queries)
    test_ids = set(q["query_id"] for q in test_queries)

    assert train_ids.isdisjoint(test_ids), "[ERROR] Train và Test bị overlap."
    assert train_ids | test_ids == all_ids, "[ERROR] Train và Test không bao phủ toàn bộ query."
    print("\n[XÁC NHẬN] Phân chia tập chuẩn: Train ∩ Test = ∅ và Train ∪ Test = All.")

    # 5. Ghi artifact thông qua temporary files và atomic replace
    tmp_train = TRAIN_OUTPUT.with_suffix(".json.tmp")
    tmp_test = TEST_OUTPUT.with_suffix(".json.tmp")

    try:
        with open(tmp_train, "w", encoding="utf-8") as f:
            json.dump(train_queries, f, ensure_ascii=False, indent=2)

        with open(tmp_test, "w", encoding="utf-8") as f:
            json.dump(test_queries, f, ensure_ascii=False, indent=2)

        tmp_train.replace(TRAIN_OUTPUT)
        tmp_test.replace(TEST_OUTPUT)

        print(f"\nĐã lưu artifact an toàn:")
        print(f"  - Train: {TRAIN_OUTPUT}")
        print(f"  - Test : {TEST_OUTPUT}")
    finally:
        if tmp_train.exists():
            tmp_train.unlink(missing_ok=True)
        if tmp_test.exists():
            tmp_test.unlink(missing_ok=True)

    print("=" * 60)
    print("HOÀN THÀNH XÂY DỰNG RETRIEVAL BENCHMARK.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Xây dựng Retrieval Benchmark (Train/Test Split)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Buộc tạo lại benchmark ngay cả khi file đã tồn tại.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Tỷ lệ tập test (mặc định: 0.20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed cho split (mặc định: 42).",
    )
    args = parser.parse_args()
    build_retrieval_benchmark(
        force=args.force, test_size=args.test_size, seed=args.seed
    )