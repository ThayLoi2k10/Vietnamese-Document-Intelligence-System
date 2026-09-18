# scripts/validation/validate_legal_qa.py
"""Script kiểm định chất lượng artifact legal_qa_preprocessed.parquet.

Kiểm tra:
    1. Kiểm tra tồn tại dữ liệu và bảo vệ chống rỗng
    2. Kiểm định tính toàn vẹn của schema trước khi truy cập cột (chống KeyError)
    3. Kiểm định kiểu dữ liệu các cột định danh (qa_id, doc_id) và generation_time
    4. Tổng số lượng bản ghi (29,145), tính duy nhất và liên tục của qa_id
    5. Kiểm tra không có câu hỏi, câu trả lời, ngữ cảnh rỗng
    6. Chuẩn hóa 100% Unicode NFC, sạch ký tự NBSP và CR/CRLF
    7. Tính toàn vẹn và không null của taxonomy: difficulty và question_type
"""

import argparse
from pathlib import Path
import sys
from typing import List
import unicodedata
import pandas as pd

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
DEFAULT_INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_qa"
    / "legal_qa_preprocessed.parquet"
)
DEFAULT_REPORT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_qa"
    / "validation_report.txt"
)

EXPECTED_COLUMNS = [
    "qa_id",
    "doc_id",
    "doc_name",
    "doc_type_name",
    "article_content",
    "question",
    "answer",
    "difficulty",
    "question_type",
    "generation_time",
]

VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_QUESTION_TYPES = {"factual", "interpretation", "application", "analytical"}


class QAValidator:
    """Quản lý quá trình kiểm định và ghi nhận báo cáo cho Legal QA."""

    def __init__(self, report_path: Path):
        self.report_path = report_path
        self.logs: List[str] = []
        self.has_failed = False

    def log(self, message: str = "") -> None:
        print(message)
        self.logs.append(message)

    def check(self, condition: bool, name: str, detail: str = "") -> bool:
        status = "PASS" if condition else "FAIL"
        line = f"[{status}] {name}"
        self.log(line)
        if detail:
            self.log(f"       {detail}")
        if not condition:
            self.has_failed = True
        return condition

    def save_report(self) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.logs) + "\n")


def validate_legal_qa(input_file: Path, report_file: Path) -> int:
    """Thực thi toàn bộ các bài kiểm tra validation có kiểm soát ngoại lệ."""
    if not input_file.exists():
        print(f"Lỗi: Không tìm thấy file cần kiểm định tại: {input_file}")
        return 1

    v = QAValidator(report_file)
    v.log("=" * 80)
    v.log("VALIDATION: PREPROCESSED VIETNAMESE LEGAL QA")
    v.log("=" * 80)

    # 1. NẠP DỮ LIỆU & KIỂM TRA RỖNG
    v.log("\n[1] Loading data...")
    df = pd.read_parquet(input_file)
    v.log(f"Tổng số bản ghi QA đã xử lý: {len(df):,}")

    if not v.check(len(df) > 0, "Processed QA dataset is not empty", f"Actual rows: {len(df)}"):
        v.log("\n" + "=" * 80)
        v.log("RESULT: LEGAL QA VALIDATION FAILED (Empty Dataset)")
        v.log("=" * 80)
        v.save_report()
        return 1

    # 2. KIỂM ĐỊNH SCHEMA (Guard KeyError)
    v.log("\n[2] Schema validation")
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    schema_ok = len(missing) == 0
    v.check(schema_ok, "Required columns exist", f"Missing: {missing}")

    if not schema_ok:
        v.log("\n" + "=" * 80)
        v.log("RESULT: LEGAL QA VALIDATION FAILED (Missing Required Columns)")
        v.log("=" * 80)
        v.save_report()
        return 1

    # Kiểm tra kiểu dữ liệu sau khi đảm bảo schema tồn tại
    v.check(
        pd.api.types.is_integer_dtype(df["qa_id"]),
        "Column 'qa_id' has integer dtype",
        f"Actual: {df['qa_id'].dtype}",
    )
    v.check(
        pd.api.types.is_integer_dtype(df["doc_id"]),
        "Column 'doc_id' has integer dtype",
        f"Actual: {df['doc_id'].dtype}",
    )
    v.check(
        pd.api.types.is_numeric_dtype(df["generation_time"]),
        "Column 'generation_time' has numeric dtype",
        f"Actual: {df['generation_time'].dtype}",
    )

    # 3. SỐ LƯỢNG BẢN GHI VÀ TÍNH DUY NHẤT / LIÊN TỤC CỦA QA ID
    v.log("\n[3] Row count & QA ID continuity")
    v.check(len(df) == 29145, "Row count equals 29,145", f"Actual: {len(df):,}")
    v.check(df["qa_id"].notna().all(), "No missing qa_id")
    v.check(df["qa_id"].is_unique, "qa_id is unique")

    is_continuous = df["qa_id"].tolist() == list(range(len(df)))
    v.check(
        is_continuous,
        "qa_id is strictly continuous from 0 to len-1",
        f"Range: {df['qa_id'].min()} -> {df['qa_id'].max()}",
    )

    # 4. TÍNH ĐẦY ĐỦ CỦA NỘI DUNG (KHÔNG CHỨA TEXT RỖNG)
    v.log("\n[4] Text completeness validation")
    empty_q = int((df["question"].fillna("").astype(str).str.strip() == "").sum())
    empty_a = int((df["answer"].fillna("").astype(str).str.strip() == "").sum())
    empty_c = int((df["article_content"].fillna("").astype(str).str.strip() == "").sum())

    v.check(empty_q == 0, "No empty questions", f"Empty count: {empty_q}")
    v.check(empty_a == 0, "No empty answers", f"Empty count: {empty_a}")
    v.check(empty_c == 0, "No empty contexts", f"Empty count: {empty_c}")

    # 5. UNICODE NFC & KÝ TỰ RÁC
    v.log("\n[5] Unicode NFC & Artifacts")
    for col in ["question", "answer", "article_content"]:
        series = df[col].fillna("").astype(str)
        non_nfc = int(series.apply(lambda t: unicodedata.normalize("NFC", t) != t).sum())
        has_nbsp = int(series.apply(lambda t: "\xa0" in t).sum())
        has_cr = int(series.apply(lambda t: "\r" in t).sum())

        v.check(non_nfc == 0, f"Column '{col}' is 100% NFC normalized", f"Non-NFC: {non_nfc}")
        v.check(has_nbsp == 0, f"Column '{col}' contains no NBSP", f"NBSP count: {has_nbsp}")
        v.check(has_cr == 0, f"Column '{col}' contains no CR/CRLF", f"CR count: {has_cr}")

    # 6. TÍNH TOÀN VẸN CỦA TAXONOMY (CHẶN HOÀN TOÀN NULL VÀ GIÁ TRỊ NGOÀI TAXONOMY)
    v.log("\n[6] Taxonomy integrity")
    difficulty_valid = df["difficulty"].notna() & df["difficulty"].isin(VALID_DIFFICULTIES)
    v.check(
        bool(difficulty_valid.all()),
        "All difficulty values are valid and non-null",
        f"Invalid/null count: {int((~difficulty_valid).sum())}",
    )

    question_type_valid = df["question_type"].notna() & df["question_type"].isin(VALID_QUESTION_TYPES)
    v.check(
        bool(question_type_valid.all()),
        "All question_type values are valid and non-null",
        f"Invalid/null count: {int((~question_type_valid).sum())}",
    )

    # TỔNG KẾT
    v.log("\n" + "=" * 80)
    if not v.has_failed:
        v.log("RESULT: LEGAL QA VALIDATION PASSED")
        verdict = 0
    else:
        v.log("RESULT: LEGAL QA VALIDATION FAILED")
        verdict = 1
    v.log("=" * 80)

    v.save_report()
    v.log(f"Báo cáo kiểm định đã được xuất tại: {report_file}")
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kiểm định artifact legal_qa_preprocessed.parquet."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Đường dẫn tới file parquet đã tiền xử lý.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_FILE,
        help="Đường dẫn lưu file báo cáo kết quả kiểm định.",
    )
    args = parser.parse_args()
    exit_code = validate_legal_qa(input_file=args.input, report_file=args.report)
    sys.exit(exit_code)