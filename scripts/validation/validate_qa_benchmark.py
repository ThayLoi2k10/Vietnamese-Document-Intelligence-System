# scripts/validation/validate_qa_benchmark.py
"""Script kiểm định tính toàn vẹn và chất lượng của QA Benchmark Dataset (Train / Val / Test).

Kiểm tra:
    1. Kiểm tra tồn tại dữ liệu và schema 8 cột chuẩn của từng split
    2. Kiểm tra kiểu dữ liệu (Dtypes): qa_id, doc_id là integer
    3. Kiểm tra tính toàn vẹn văn bản: không null, không chuỗi rỗng
    4. Xác thực taxonomy: difficulty và question_type tuân thủ 100% danh mục chuẩn
    5. Kiểm định tính phân hoạch:
       - Rời rạc ID: qa_id duy nhất toàn cục
       - Zero Context Leakage: Train ∩ Val ∩ Test = ∅ trên content_hash
       - Bảo toàn dữ liệu: Train ∪ Val ∪ Test = Source All QA
       - Tỷ lệ phân tập: ~70% Train, ~15% Val, ~15% Test
    6. Xuất báo cáo kiểm định và trả về mã thoát cho CI/CD
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple
import pandas as pd
import pyarrow.parquet as pq

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
DEFAULT_BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmarks"
DEFAULT_TRAIN_PATH = DEFAULT_BENCHMARK_DIR / "qa_train.parquet"
DEFAULT_VAL_PATH = DEFAULT_BENCHMARK_DIR / "qa_val.parquet"
DEFAULT_TEST_PATH = DEFAULT_BENCHMARK_DIR / "qa_test.parquet"
DEFAULT_SOURCE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "legal_qa" / "legal_qa_preprocessed.parquet"
)
DEFAULT_REPORT_PATH = DEFAULT_BENCHMARK_DIR / "qa_validation_report.txt"

EXPECTED_COLUMNS = [
    "qa_id",
    "doc_id",
    "content_hash",
    "question",
    "answer",
    "context",
    "difficulty",
    "question_type",
]

VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_QUESTION_TYPES = {"factual", "interpretation", "application", "analytical"}


def validate_single_split(df: pd.DataFrame, split_name: str, logs: List[str]) -> bool:
    """Kiểm tra schema, kiểu dữ liệu, tính toàn vẹn nội bộ của một tập split."""
    logs.append(f"\n--- KIỂM ĐỊNH TẬP: {split_name} ({len(df):,} bản ghi) ---")
    if len(df) == 0:
        logs.append(f"[FAIL] Tập {split_name} rỗng bản ghi!")
        return False

    is_valid = True

    # 1. Kiểm tra cấu trúc cột (Fail fast nếu sai schema để bảo vệ các check phía sau)
    if list(df.columns) != EXPECTED_COLUMNS:
        logs.append(
            f"[FAIL] Sai lệch cấu trúc cột! Thực tế: {list(df.columns)}, Kỳ vọng: {EXPECTED_COLUMNS}"
        )
        return False
    else:
        logs.append("[PASS] Đúng cấu trúc 8 cột chuẩn.")

    # 2. Kiểm tra kiểu dữ liệu (Dtypes)
    if not pd.api.types.is_integer_dtype(df["qa_id"]):
        logs.append(f"[FAIL] Cột 'qa_id' không phải integer dtype: {df['qa_id'].dtype}")
        is_valid = False
    if not pd.api.types.is_integer_dtype(df["doc_id"]):
        logs.append(f"[FAIL] Cột 'doc_id' không phải integer dtype: {df['doc_id'].dtype}")
        is_valid = False

    # 3. Kiểm tra Null
    null_counts = df.isnull().sum().to_dict()
    has_null = any(v > 0 for v in null_counts.values())
    if has_null:
        logs.append(f"[FAIL] Có giá trị null trong tập {split_name}: {null_counts}")
        is_valid = False
    else:
        logs.append("[PASS] 0 giá trị null.")

    # 4. Kiểm tra chuỗi rỗng
    text_cols = ["question", "answer", "context", "content_hash"]
    empty_counts = {
        col: int((df[col].astype(str).str.strip() == "").sum()) for col in text_cols
    }
    has_empty = any(v > 0 for v in empty_counts.values())
    if has_empty:
        logs.append(f"[FAIL] Phát hiện text rỗng trong tập {split_name}: {empty_counts}")
        is_valid = False
    else:
        logs.append("[PASS] 0 text rỗng sau chuẩn hóa.")

    # 5. Kiểm tra tính toàn vẹn của Taxonomy
    diff_valid = df["difficulty"].isin(VALID_DIFFICULTIES).all()
    if not diff_valid:
        invalid_diffs = set(df["difficulty"].unique()) - VALID_DIFFICULTIES
        logs.append(f"[FAIL] Phát hiện nhãn difficulty không hợp lệ: {invalid_diffs}")
        is_valid = False
    else:
        logs.append("[PASS] Nhãn difficulty hợp lệ theo taxonomy.")

    type_valid = df["question_type"].isin(VALID_QUESTION_TYPES).all()
    if not type_valid:
        invalid_types = set(df["question_type"].unique()) - VALID_QUESTION_TYPES
        logs.append(f"[FAIL] Phát hiện nhãn question_type không hợp lệ: {invalid_types}")
        is_valid = False
    else:
        logs.append("[PASS] Nhãn question_type hợp lệ theo taxonomy.")

    # 6. Kiểm tra trùng lặp qa_id nội bộ
    dup_qas = int(df["qa_id"].duplicated().sum())
    if dup_qas > 0:
        logs.append(f"[FAIL] Trùng lặp {dup_qas} qa_id trong nội bộ tập {split_name}!")
        is_valid = False
    else:
        logs.append("[PASS] Toàn bộ qa_id là duy nhất nội bộ.")

    logs.append(f"  * Số lượng contexts (unique content_hash): {df['content_hash'].nunique():,}")
    logs.append(f"  * Trạng thái kiểm tra nội bộ: {'PASS' if is_valid else 'FAIL'}")
    return is_valid


def validate_qa_benchmark(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    source_path: Path,
    report_path: Path,
) -> int:
    """Thực thi kiểm định toàn diện bộ 3 tập QA Benchmark."""
    logs: List[str] = []
    logs.append("=" * 60)
    logs.append("KIỂM ĐỊNH TOÀN VẸN: QA BENCHMARK DATASET")
    logs.append("=" * 60)

    # 1. Kiểm tra tồn tại các file artifact
    for p in [train_path, val_path, test_path]:
        if not p.exists():
            logs.append(f"[CRITICAL] Không tìm thấy file benchmark: {p}")
            print("\n".join(logs))
            return 1

    try:
        df_train = pq.read_table(train_path).to_pandas()
        df_val = pq.read_table(val_path).to_pandas()
        df_test = pq.read_table(test_path).to_pandas()
    except Exception as e:
        logs.append(f"[CRITICAL] Lỗi đọc file Parquet benchmark: {e}")
        print("\n".join(logs))
        return 1

    total_records = len(df_train) + len(df_val) + len(df_test)
    logs.append(f"Tổng số bản ghi cả 3 tập: {total_records:,}")

    # 2. Validate từng split độc lập
    train_ok = validate_single_split(df_train, "TRAIN", logs)
    val_ok = validate_single_split(df_val, "VAL", logs)
    test_ok = validate_single_split(df_test, "TEST", logs)

    # 3. Kiểm tra tính duy nhất toàn cục của qa_id
    logs.append("\n--- KIỂM TRA ĐỊNH DANH TOÀN CỤC & TÍNH RỜI RẠC ---")
    train_qids = set(df_train["qa_id"]) if "qa_id" in df_train else set()
    val_qids = set(df_val["qa_id"]) if "qa_id" in df_val else set()
    test_qids = set(df_test["qa_id"]) if "qa_id" in df_test else set()

    qid_disjoint = (
        train_qids.isdisjoint(val_qids)
        and train_qids.isdisjoint(test_qids)
        and val_qids.isdisjoint(test_qids)
    )
    if not qid_disjoint:
        logs.append("[FAIL] Trùng lặp qa_id giữa các tập phân chia!")
    else:
        logs.append("[PASS] qa_id độc lập tuyệt đối giữa Train, Val và Test.")

    # 4. Kiểm tra đối chiếu với nguồn dữ liệu gốc (Train ∪ Val ∪ Test = Source)
    source_match = True
    if source_path.exists():
        try:
            source_table = pq.read_table(source_path, columns=["qa_id"])
            source_qids = set(source_table["qa_id"].to_pylist())
            combined_qids = train_qids | val_qids | test_qids

            if combined_qids != source_qids:
                missing_from_splits = source_qids - combined_qids
                extra_in_splits = combined_qids - source_qids
                logs.append(
                    f"[FAIL] Phân tập không khớp toàn bộ dữ liệu nguồn!\n"
                    f"       Thiếu: {len(missing_from_splits)}, Dư: {len(extra_in_splits)}"
                )
                source_match = False
            else:
                logs.append(
                    f"[PASS] Train ∪ Val ∪ Test = Source All ({len(source_qids):,} bản ghi được bảo toàn 100%)."
                )
        except Exception as e:
            logs.append(f"[WARN] Không thể đọc source file để đối chiếu: {e}")
    else:
        logs.append(f"[WARN] Không tìm thấy source file {source_path.name} để đối chiếu phân hoạch.")

    # 5. Kiểm tra rò rỉ ngữ cảnh (Zero Leakage trên content_hash)
    logs.append("\n--- KIỂM TRA ĐỘC LẬP NGỮ CẢNH (ZERO LEAKAGE) ---")
    train_hashes = set(df_train["content_hash"]) if "content_hash" in df_train else set()
    val_hashes = set(df_val["content_hash"]) if "content_hash" in df_val else set()
    test_hashes = set(df_test["content_hash"]) if "content_hash" in df_test else set()

    leak_train_val = train_hashes.intersection(val_hashes)
    leak_train_test = train_hashes.intersection(test_hashes)
    leak_val_test = val_hashes.intersection(test_hashes)

    leakage_ok = (
        len(leak_train_val) == 0
        and len(leak_train_test) == 0
        and len(leak_val_test) == 0
    )

    if not leakage_ok:
        logs.append("[FAIL] Phát hiện rò rỉ ngữ cảnh giữa các tập!")
        logs.append(f"  * Train ∩ Val : {len(leak_train_val)} hashes")
        logs.append(f"  * Train ∩ Test: {len(leak_train_test)} hashes")
        logs.append(f"  * Val ∩ Test  : {len(leak_val_test)} hashes")
    else:
        logs.append("[PASS] Train ∩ Val = ∅")
        logs.append("[PASS] Train ∩ Test = ∅")
        logs.append("[PASS] Val ∩ Test = ∅")
        logs.append("[PASS] 0% context leakage (tất cả câu hỏi cùng context nằm trọn trong 1 tập duy nhất).")

    # 6. Kiểm tra tỷ lệ phân chia (~70% Train, ~15% Val, ~15% Test)
    logs.append("\n--- KIỂM TRA TỶ LỆ PHÂN TẬP ---")
    train_ratio = len(df_train) / total_records if total_records > 0 else 0
    val_ratio = len(df_val) / total_records if total_records > 0 else 0
    test_ratio = len(df_test) / total_records if total_records > 0 else 0

    ratio_ok = (
        0.67 <= train_ratio <= 0.73
        and 0.13 <= val_ratio <= 0.17
        and 0.13 <= test_ratio <= 0.17
    )
    if not ratio_ok:
        logs.append(
            f"[FAIL] Tỷ lệ phân tập lệch chuẩn (Train: {train_ratio*100:.1f}%, "
            f"Val: {val_ratio*100:.1f}%, Test: {test_ratio*100:.1f}%)"
        )
    else:
        logs.append(
            f"[PASS] Tỷ lệ chuẩn (Train: {train_ratio*100:.1f}%, "
            f"Val: {val_ratio*100:.1f}%, Test: {test_ratio*100:.1f}%)."
        )

    # 7. Phân phối Taxonomy
    logs.append("\n--- BẢNG PHÂN PHỐI TAXONOMY GIỮA CÁC TẬP (%) ---")
    logs.append("\n[Phân phối Difficulty %]:")
    diff_table = pd.DataFrame(
        {
            "Train": df_train["difficulty"].value_counts(normalize=True) * 100 if "difficulty" in df_train else {},
            "Val": df_val["difficulty"].value_counts(normalize=True) * 100 if "difficulty" in df_val else {},
            "Test": df_test["difficulty"].value_counts(normalize=True) * 100 if "difficulty" in df_test else {},
        }
    ).round(2)
    logs.append(diff_table.to_string())

    logs.append("\n[Phân phối Question Type %]:")
    qtype_table = pd.DataFrame(
        {
            "Train": df_train["question_type"].value_counts(normalize=True) * 100 if "question_type" in df_train else {},
            "Val": df_val["question_type"].value_counts(normalize=True) * 100 if "question_type" in df_val else {},
            "Test": df_test["question_type"].value_counts(normalize=True) * 100 if "question_type" in df_test else {},
        }
    ).round(2)
    logs.append(qtype_table.to_string())

    # 8. Tổng kết & Ghi báo cáo
    all_passed = (
        train_ok
        and val_ok
        and test_ok
        and qid_disjoint
        and source_match
        and leakage_ok
        and ratio_ok
    )

    logs.append("\n" + "=" * 60)
    logs.append(
        f"KẾT QUẢ CUỐI CÙNG: {'ALL ASSERTIONS PASSED (100%)' if all_passed else 'FAILED'}"
    )
    logs.append("=" * 60)

    report_content = "\n".join(logs)
    print(report_content)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content + "\n")
    print(f"\nĐã ghi báo cáo: {report_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kiểm định tính toàn vẹn của QA Benchmark Dataset (Train/Val/Test)."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="Đường dẫn file qa_train.parquet.",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=DEFAULT_VAL_PATH,
        help="Đường dẫn file qa_val.parquet.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=DEFAULT_TEST_PATH,
        help="Đường dẫn file qa_test.parquet.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help="Đường dẫn file legal_qa_preprocessed.parquet.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Đường dẫn lưu file báo cáo kiểm định.",
    )
    args = parser.parse_args()

    exit_code = validate_qa_benchmark(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        source_path=args.source,
        report_path=args.report,
    )
    sys.exit(exit_code)