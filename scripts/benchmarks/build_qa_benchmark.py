# scripts/benchmarks/build_qa_benchmark.py
"""Script xây dựng QA Benchmark (Train / Val / Test Split).

Thực hiện:
    - Nạp legal_qa_preprocessed.parquet
    - Kiểm định tính toàn vẹn của qa_id và text
    - Tạo content_hash từ article_content để nhóm (Grouping chống rò rỉ ngữ cảnh)
    - Tạo strata từ difficulty và question_type để phân tầng
    - Thực hiện StratifiedGroupKFold 2 giai đoạn (70% Train / 15% Val / 15% Test)
    - Kiểm định phân hoạch toán học nghiêm ngặt: Rời rạc và Bao phủ toàn bộ bản ghi
    - Ghi artifact thông qua temporary files và atomic replace

Input:  data/processed/legal_qa/legal_qa_preprocessed.parquet
Output: data/benchmarks/qa_train.parquet
        data/benchmarks/qa_val.parquet
        data/benchmarks/qa_test.parquet
"""

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any
import unicodedata
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedGroupKFold

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
INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "legal_qa" / "legal_qa_preprocessed.parquet"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "benchmarks"
TRAIN_OUTPUT = OUTPUT_DIR / "qa_train.parquet"
VAL_OUTPUT = OUTPUT_DIR / "qa_val.parquet"
TEST_OUTPUT = OUTPUT_DIR / "qa_test.parquet"

TARGET_COLUMNS = [
    "qa_id",
    "doc_id",
    "content_hash",
    "question",
    "answer",
    "context",
    "difficulty",
    "question_type",
]


def clean_text(text: Any) -> str:
    """Chuẩn hóa Unicode NFC và gộp khoảng trắng thừa."""
    if not isinstance(text, str) or not text.strip():
        return ""
    normalized = unicodedata.normalize("NFC", text)
    return " ".join(normalized.split())


def compute_hash(text: str) -> str:
    """Tạo mã băm MD5 đại diện cho ngữ cảnh văn bản."""
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def build_qa_benchmark(force: bool = False, seed: int = 42) -> None:
    """Thực thi phân chia tập benchmark QA với Stratified Group Split."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file nguồn: {INPUT_PATH}. "
            "Hãy chạy scripts/processing/preprocess_legal_qa.py trước."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Kiểm tra artifact đã tồn tại
    artifacts = [TRAIN_OUTPUT, VAL_OUTPUT, TEST_OUTPUT]
    if not force and all(p.exists() and p.stat().st_size > 0 for p in artifacts):
        print(
            "Artifacts benchmark QA đã tồn tại đầy đủ:\n"
            f"  - {TRAIN_OUTPUT}\n"
            f"  - {VAL_OUTPUT}\n"
            f"  - {TEST_OUTPUT}\n"
            "Bỏ qua xử lý (Dùng --force để tạo lại)."
        )
        return

    print("=" * 60)
    print("XÂY DỰNG QA BENCHMARK (STRATIFIED GROUP SPLIT)")
    print("=" * 60)

    print(f"Đọc dữ liệu: {INPUT_PATH.name}...")
    df = pq.read_table(INPUT_PATH).to_pandas()
    total_records = len(df)
    print(f"Tổng số bản ghi QA nạp vào: {total_records:,}")

    if total_records == 0:
        raise ValueError("Tập dữ liệu đầu vào rỗng, không thể tạo benchmark.")

    # 1. Validation sơ bộ định danh qa_id
    if df["qa_id"].isna().any():
        raise ValueError("Phát hiện giá trị qa_id rỗng trong dữ liệu đầu vào.")
    if not df["qa_id"].is_unique:
        raise ValueError("Phát hiện qa_id bị trùng lặp trong dữ liệu đầu vào.")

    # 2. Tiền xử lý văn bản & tạo khóa phân nhóm chống leakage
    print("\nChuẩn hóa text và tạo content_hash...")
    df["question"] = df["question"].apply(clean_text)
    df["answer"] = df["answer"].apply(clean_text)
    df["context"] = df["article_content"].apply(clean_text)

    # Fail-fast nếu text rỗng sau clean
    if (df["question"] == "").any():
        raise ValueError("Phát hiện question rỗng sau khi clean_text.")
    if (df["answer"] == "").any():
        raise ValueError("Phát hiện answer rỗng sau khi clean_text.")
    if (df["context"] == "").any():
        raise ValueError("Phát hiện context rỗng sau khi clean_text.")

    df["content_hash"] = df["context"].apply(compute_hash)
    df["strata"] = df["difficulty"].astype(str) + "_" + df["question_type"].astype(str)

    # 3. Stratified Group Split (70% Train / 15% Val / 15% Test)
    print("\nThực hiện Stratified Group Split theo content_hash...")
    groups = df["content_hash"].values
    y = df["strata"].values
    X = np.zeros(total_records)

    # Giai đoạn 1: 10 folds (7 folds Train ~ 70%, 3 folds Temp ~ 30%)
    sgkf_10 = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    fold_assignments = np.zeros(total_records, dtype=int)
    for fold, (_, val_idx) in enumerate(sgkf_10.split(X, y, groups)):
        fold_assignments[val_idx] = fold

    train_mask = fold_assignments >= 3
    temp_mask = fold_assignments < 3

    df_train = df[train_mask].copy()
    df_temp = df[temp_mask].copy()

    # Giai đoạn 2: Chia đôi Temp thành Val (15%) và Test (15%)
    sgkf_2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    temp_groups = df_temp["content_hash"].values
    temp_y = df_temp["strata"].values
    temp_X = np.zeros(len(df_temp))

    val_idx_sub, test_idx_sub = next(sgkf_2.split(temp_X, temp_y, temp_groups))

    df_val = df_temp.iloc[val_idx_sub].copy()
    df_test = df_temp.iloc[test_idx_sub].copy()

    # 4. Thống kê kết quả phân tập
    print("\n" + "-" * 20 + " THỐNG KÊ PHÂN TẬP " + "-" * 20)
    print(
        f"Train: {len(df_train):,} QA ({len(df_train)/total_records*100:.2f}%) | "
        f"{df_train['content_hash'].nunique():,} contexts"
    )
    print(
        f"Val  : {len(df_val):,} QA ({len(df_val)/total_records*100:.2f}%) | "
        f"{df_val['content_hash'].nunique():,} contexts"
    )
    print(
        f"Test : {len(df_test):,} QA ({len(df_test)/total_records*100:.2f}%) | "
        f"{df_test['content_hash'].nunique():,} contexts"
    )

    # 5. Kiểm tra ranh giới ngữ cảnh (Zero Context Leakage)
    train_hashes = set(df_train["content_hash"])
    val_hashes = set(df_val["content_hash"])
    test_hashes = set(df_test["content_hash"])

    assert train_hashes.isdisjoint(val_hashes), "[ERROR] Rò rỉ context giữa Train và Val!"
    assert train_hashes.isdisjoint(test_hashes), "[ERROR] Rò rỉ context giữa Train và Test!"
    assert val_hashes.isdisjoint(test_hashes), "[ERROR] Rò rỉ context giữa Val và Test!"
    print("\n[XÁC NHẬN ZERO LEAKAGE] Độc lập tuyệt đối về context (Train ∩ Val ∩ Test = ∅).")

    # 6. Kiểm tra tính phân hoạch toàn vẹn (Train ∪ Val ∪ Test = All)
    all_qa_ids = set(df["qa_id"])
    train_ids = set(df_train["qa_id"])
    val_ids = set(df_val["qa_id"])
    test_ids = set(df_test["qa_id"])

    assert train_ids.isdisjoint(val_ids), "[ERROR] qa_id bị trùng giữa Train và Val!"
    assert train_ids.isdisjoint(test_ids), "[ERROR] qa_id bị trùng giữa Train và Test!"
    assert val_ids.isdisjoint(test_ids), "[ERROR] qa_id bị trùng giữa Val và Test!"
    assert (train_ids | val_ids | test_ids) == all_qa_ids, (
        f"[ERROR] Tổng các tập không khớp toàn bộ dataset ban đầu! "
        f"Missing: {len(all_qa_ids - (train_ids | val_ids | test_ids))}"
    )
    print("[XÁC NHẬN BẢO TOÀN] Phân hoạch hoàn hảo: Train ∪ Val ∪ Test = All.")

    # 7. Khóa schema cột đích
    df_train = df_train[TARGET_COLUMNS]
    df_val = df_val[TARGET_COLUMNS]
    df_test = df_test[TARGET_COLUMNS]

    # 8. Ghi artifacts qua temporary files và atomic replace
    print("\nĐang ghi artifacts vào data/benchmarks/...")
    tmp_train = TRAIN_OUTPUT.with_suffix(".parquet.tmp")
    tmp_val = VAL_OUTPUT.with_suffix(".parquet.tmp")
    tmp_test = TEST_OUTPUT.with_suffix(".parquet.tmp")

    try:
        pq.write_table(pa.Table.from_pandas(df_train, preserve_index=False), str(tmp_train), compression="snappy")
        pq.write_table(pa.Table.from_pandas(df_val, preserve_index=False), str(tmp_val), compression="snappy")
        pq.write_table(pa.Table.from_pandas(df_test, preserve_index=False), str(tmp_test), compression="snappy")

        tmp_train.replace(TRAIN_OUTPUT)
        tmp_val.replace(VAL_OUTPUT)
        tmp_test.replace(TEST_OUTPUT)

        print(f"  - Đã lưu: {TRAIN_OUTPUT.name}")
        print(f"  - Đã lưu: {VAL_OUTPUT.name}")
        print(f"  - Đã lưu: {TEST_OUTPUT.name}")
    finally:
        for tmp_file in [tmp_train, tmp_val, tmp_test]:
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)

    print("=" * 60)
    print("HOÀN THÀNH XÂY DỰNG QA BENCHMARK.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Xây dựng QA Benchmark (Train/Val/Test Split với Group Stratification)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Buộc chạy lại phân chia benchmark ngay cả khi file đã tồn tại.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed phục vụ reproducibility (mặc định: 42).",
    )
    args = parser.parse_args()
    build_qa_benchmark(force=args.force, seed=args.seed)