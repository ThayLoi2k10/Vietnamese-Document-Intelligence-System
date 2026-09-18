# scripts/benchmarks/audit_evaluation_sources.py
"""Script kiểm toán cấu trúc, semantic role và tính toàn vẹn của các nguồn dữ liệu Evaluation.

Kiểm tra:
    1. Retrieval JSON sources (train.json, public_test.json, private_test.json):
       - Quét 100% bản ghi để xác thực invariant (qid hợp lệ/duy nhất, text câu hỏi tồn tại)
       - Kiểm tra tính nhất quán của relevant_laws (kiểu list of int) trên toàn bộ tập dữ liệu
       - Phân định rõ ngữ nghĩa (semantics): Ground truth benchmark vs Blind test
    2. QA Parquet (legal_qa_preprocessed.parquet):
       - Kiểm định schema tối thiểu bắt buộc phục vụ downstream benchmark/evaluation
       - Phân tích độ phủ, missing values, empty strings và cardinality
    3. Trả về exit code chuẩn cho CI/CD (0 nếu các nguồn bắt buộc hợp lệ, 1 nếu vi phạm)
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, List
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
RAW_RETRIEVAL_DIR = PROJECT_ROOT / "data" / "raw" / "legal_retrieval"
PROCESSED_QA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "legal_qa" / "legal_qa_preprocessed.parquet"
)

# Schema tối thiểu bắt buộc từ legal_qa_preprocessed để build QA benchmark
REQUIRED_QA_COLUMNS = {
    "qa_id",
    "doc_id",
    "article_content",
    "question",
    "answer",
    "difficulty",
    "question_type",
}


def inspect_retrieval_json(
    file_path: Path,
    name: str,
    expected_role: str,
    required: bool = True,
) -> bool:
    """Kiểm toán toàn bộ bản ghi của file JSON retrieval và xác định vai trò dữ liệu."""
    print(f"\n{'=' * 25} AUDIT: {name} {'=' * 25}")
    print(f"Vai trò kỳ vọng: {expected_role}")

    if not file_path.exists():
        tag = "[ERROR - REQUIRED]" if required else "[WARN - OPTIONAL]"
        print(f"{tag} File không tồn tại: {file_path}")
        return not required

    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"Đường dẫn: {file_path}")
    print(f"Dung lượng: {file_size_mb:.2f} MB")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[FAIL] Lỗi parse JSON: {e}")
        return False

    records: List[Any] = []
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                records = v
                break

    if not records:
        print("[FAIL] File rỗng hoặc không chứa danh sách bản ghi nào.")
        return not required

    total_records = len(records)
    print(f"Tổng số bản ghi: {total_records:,}")

    # 1. Quét toàn bộ bản ghi kiểm tra invariants (Invariants check across all records)
    invalid_record_types = 0
    missing_qid_count = 0
    empty_question_count = 0
    qids = []

    has_relevant_laws_key = 0
    non_list_laws_count = 0
    non_int_laws_count = 0
    labeled_queries_count = 0

    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            invalid_record_types += 1
            continue

        qid = item.get("qid") if "qid" in item else item.get("id")
        if qid is None:
            missing_qid_count += 1
        else:
            qids.append(qid)

        q_text = item.get("question")
        if not isinstance(q_text, str) or not q_text.strip():
            empty_question_count += 1

        if "relevant_laws" in item:
            has_relevant_laws_key += 1
            laws = item["relevant_laws"]
            if not isinstance(laws, list):
                non_list_laws_count += 1
            else:
                if any(not isinstance(doc_id, int) for doc_id in laws):
                    non_int_laws_count += 1
                if len(laws) > 0:
                    labeled_queries_count += 1

    duplicate_qids = len(qids) - len(set(qids))

    print("\n--- Kiểm tra Invariants (Toàn bộ bản ghi) ---")
    print(f"• Bản ghi không phải dict        : {invalid_record_types}")
    print(f"• Bản ghi thiếu trường 'qid'/'id': {missing_qid_count}")
    print(f"• Bản ghi trùng lặp 'qid'/'id'   : {duplicate_qids}")
    print(f"• Bản ghi có question rỗng       : {empty_question_count}")
    print(f"• Bản ghi có trường 'relevant_laws': {has_relevant_laws_key:,} / {total_records:,}")
    print(f"• Trường relevant_laws sai kiểu (non-list): {non_list_laws_count}")
    print(f"• Chứa ID không phải số nguyên (non-int) : {non_int_laws_count}")
    print(f"• Số query có nhãn ground truth (>0 ID) : {labeled_queries_count:,} / {total_records:,}")

    # 2. Đánh giá ngữ nghĩa nguồn dữ liệu (Semantic Classification)
    print("\n--- Phân định Ngữ nghĩa (Semantics & Usage) ---")
    is_valid_ground_truth = (
        has_relevant_laws_key == total_records
        and non_list_laws_count == 0
        and non_int_laws_count == 0
        and labeled_queries_count == total_records
    )

    if is_valid_ground_truth:
        print(
            "-> KẾT LUẬN: ĐẦY ĐỦ GROUND TRUTH (100% labeled).\n"
            "   Ý nghĩa: Dùng làm nguồn xây dựng retrieval benchmark (Train/Test split) "
            "   để đo Recall@k và MRR cục bộ."
        )
    elif labeled_queries_count == 0:
        print(
            "-> KẾT LUẬN: BLIND TEST SET (0% labeled / nhãn rỗng).\n"
            "   Ý nghĩa: Tập query đánh giá ẩn từ ban tổ chức. KHÔNG dùng làm ground truth "
            "   để tính toán retrieval metrics nội bộ."
        )
    else:
        print(
            f"-> KẾT LUẬN: PARTIALLY LABELED ({labeled_queries_count / total_records * 100:.1f}% labeled).\n"
            "   Ý nghĩa: Cần lọc bỏ các query không nhãn trước khi đưa vào benchmark."
        )

    # 3. Phán quyết tính hợp lệ cho pipeline
    if invalid_record_types > 0 or missing_qid_count > 0 or duplicate_qids > 0:
        print(f"[FAIL] {name} vi phạm tính toàn vẹn của cấu trúc định danh ('qid'/'id').")
        return False

    if required and not is_valid_ground_truth:
        print(f"[FAIL] {name} là file bắt buộc làm ground truth nhưng thiếu/lỗi nhãn.")
        return False

    print(f"[PASS] Cấu trúc {name} hợp lệ theo vai trò đã xác định.")
    return True


def inspect_parquet_qa(file_path: Path, required: bool = True) -> bool:
    """Kiểm toán schema tối thiểu và profiling dữ liệu QA Parquet."""
    print(f"\n{'=' * 25} AUDIT: {file_path.name} {'=' * 25}")
    if not file_path.exists():
        tag = "[ERROR - REQUIRED]" if required else "[WARN - OPTIONAL]"
        print(f"{tag} File không tồn tại: {file_path}")
        return not required

    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"Đường dẫn: {file_path}")
    print(f"Dung lượng: {file_size_mb:.2f} MB")

    try:
        table = pq.read_table(file_path)
        df = table.to_pandas()
    except Exception as e:
        print(f"[FAIL] Không thể đọc file Parquet: {e}")
        return False

    print(f"Số dòng (rows): {len(df):,}")
    print(f"Số cột (columns): {len(df.columns)}")

    if len(df) == 0:
        print("[FAIL] File Parquet rỗng bản ghi.")
        return False

    # 1. Kiểm tra Schema tối thiểu bắt buộc
    print("\n--- Kiểm tra Schema tối thiểu (Required Columns) ---")
    actual_columns = set(df.columns)
    missing_cols = REQUIRED_QA_COLUMNS - actual_columns
    if missing_cols:
        print(f"[FAIL] Thiếu các cột bắt buộc cho downstream benchmark: {missing_cols}")
        return False
    print(f"[PASS] Có đủ {len(REQUIRED_QA_COLUMNS)} cột bắt buộc: {sorted(list(REQUIRED_QA_COLUMNS))}")

    # 2. Chi tiết Schema & Profiling
    print("\n--- Chi tiết Cột & Thống kê Missing ---")
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        is_str_col = df[col].dtype == "object" or str(df[col].dtype).startswith("string")
        empty_str_count = int((df[col] == "").sum()) if is_str_col else 0
        unique_count = int(df[col].nunique())
        print(
            f"  * {col:<20} | Dtype: {str(df[col].dtype):<10} | "
            f"Nulls: {null_count:<6} | Empty: {empty_str_count:<6} | Unique: {unique_count:,}"
        )

    # 3. Trích đoạn bản ghi mẫu
    print("\n--- Bản ghi mẫu 1 (trích đoạn) ---")
    sample_dict = df.iloc[0].to_dict()
    for k in sorted(sample_dict.keys()):
        v_str = str(sample_dict[k])
        if len(v_str) > 100:
            v_str = v_str[:100] + "..."
        print(f"  {k:<20}: {v_str}")

    return True


def audit_evaluation_sources(
    retrieval_dir: Path = RAW_RETRIEVAL_DIR,
    qa_path: Path = PROCESSED_QA_PATH,
) -> int:
    """Thực thi kiểm toán và phân định semantic role cho toàn bộ nguồn evaluation."""
    print("=" * 60)
    print("KIỂM TOÁN CẤU TRÚC VÀ VAI TRÒ CÁC NGUỒN EVALUATION")
    print("=" * 60)

    success = True

    # 1. train.json: Nguồn benchmark retrieval chính (Bắt buộc phải có nhãn)
    train_ok = inspect_retrieval_json(
        retrieval_dir / "train.json",
        "train.json",
        expected_role="Benchmark Ground Truth Source (Bắt buộc có labels)",
        required=True,
    )

    # 2. public_test.json: Đánh giá public (Tùy chọn, kiểm tra nhãn)
    public_ok = inspect_retrieval_json(
        retrieval_dir / "public_test.json",
        "public_test.json",
        expected_role="Public Evaluation / Test Queries",
        required=False,
    )

    # 3. private_test.json: Blind evaluation (Tùy chọn, thường không nhãn)
    private_ok = inspect_retrieval_json(
        retrieval_dir / "private_test.json",
        "private_test.json",
        expected_role="Private Blind Test Queries (Không dùng làm GT)",
        required=False,
    )

    # 4. legal_qa_preprocessed.parquet: Nguồn cho QA benchmark (Bắt buộc)
    qa_ok = inspect_parquet_qa(qa_path, required=True)

    if not (train_ok and public_ok and private_ok and qa_ok):
        success = False

    print(f"\n{'=' * 60}")
    if success:
        print("KẾT QUẢ: TẤT CẢ NGUỒN EVALUATION ĐẠT TIÊU CHUẨN XÁC THỰC.")
        return 0
    else:
        print("KẾT QUẢ: PHÁT HIỆN VI PHẠM CẤU TRÚC HOẶC THIẾU DỮ LIỆU BẮT BUỘC.")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kiểm toán cấu trúc và vai trò các nguồn dữ liệu evaluation."
    )
    parser.add_argument(
        "--retrieval-dir",
        type=Path,
        default=RAW_RETRIEVAL_DIR,
        help="Thư mục chứa các file json của legal_retrieval.",
    )
    parser.add_argument(
        "--qa-path",
        type=Path,
        default=PROCESSED_QA_PATH,
        help="Đường dẫn file legal_qa_preprocessed.parquet.",
    )
    args = parser.parse_args()

    exit_code = audit_evaluation_sources(
        retrieval_dir=args.retrieval_dir,
        qa_path=args.qa_path,
    )
    sys.exit(exit_code)