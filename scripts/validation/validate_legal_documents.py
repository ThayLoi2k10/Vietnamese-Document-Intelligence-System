# scripts/validation/validate_legal_documents.py
"""Script kiểm định chất lượng artifact legal_documents_preprocessed.parquet.

Kiểm tra:
    1. Kiểm tra tồn tại dữ liệu và bảo vệ chống ZeroDivision
    2. Kiểm định tính toàn vẹn của schema trước khi truy cập cột (chống KeyError)
    3. Kiểu dữ liệu (Data Types) các trường số và cờ boolean
    4. Tổng số lượng bản ghi (170,824) và tính duy nhất của Document ID
    5. Tính nhất quán của cờ is_empty và char_count
    6. Mức độ bao phủ thực tế của Metadata (>95% đối với title, loại trừ chuỗi rỗng)
    7. Chất lượng văn bản: Chuẩn hóa Unicode NFC, sạch ký tự NBSP, CR và tàn dư thẻ HTML
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
    / "legal_documents"
    / "legal_documents_preprocessed.parquet"
)
DEFAULT_REPORT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_documents"
    / "validation_report.txt"
)

EXPECTED_COLS = [
    "id",
    "title",
    "so_ky_hieu",
    "loai_van_ban",
    "co_quan_ban_hanh",
    "ngay_ban_hanh",
    "ngay_co_hieu_luc",
    "tinh_trang_hieu_luc",
    "linh_vuc",
    "text",
    "char_count",
    "word_count",
    "is_empty",
]


class DocValidator:
    """Quản lý quá trình kiểm định và ghi nhận báo cáo."""

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


def validate_legal_documents(input_file: Path, report_file: Path) -> int:
    """Thực thi toàn bộ các bài kiểm tra validation có kiểm soát ngoại lệ."""
    if not input_file.exists():
        print(f"Lỗi: Không tìm thấy file cần kiểm định tại: {input_file}")
        return 1

    v = DocValidator(report_file)
    v.log("=" * 80)
    v.log("VALIDATION: PREPROCESSED LEGAL DOCUMENTS")
    v.log("=" * 80)

    # 1. NẠP DỮ LIỆU & KIỂM TRA RỖNG (Guard ZeroDivisionError)
    v.log("\n[1] Loading data...")
    df = pd.read_parquet(input_file)
    v.log(f"Tổng số văn bản đã xử lý: {len(df):,}")

    if not v.check(len(df) > 0, "Processed dataset is not empty", f"Actual rows: {len(df)}"):
        v.log("\n" + "=" * 80)
        v.log("RESULT: LEGAL DOCUMENTS VALIDATION FAILED (Empty Dataset)")
        v.log("=" * 80)
        v.save_report()
        return 1

    # 2. KIỂM ĐỊNH SCHEMA (Guard KeyError)
    v.log("\n[2] Schema validation")
    missing_cols = set(EXPECTED_COLS) - set(df.columns)
    schema_ok = len(missing_cols) == 0
    v.check(schema_ok, "Required columns exist", f"Missing: {missing_cols}")

    if not schema_ok:
        v.log("\n" + "=" * 80)
        v.log("RESULT: LEGAL DOCUMENTS VALIDATION FAILED (Missing Required Columns)")
        v.log("=" * 80)
        v.save_report()
        return 1

    # Kiểm định Data Types sau khi chắc chắn các cột tồn tại
    v.check(
        pd.api.types.is_integer_dtype(df["char_count"]),
        "Column 'char_count' has integer dtype",
        f"Actual: {df['char_count'].dtype}",
    )
    v.check(
        pd.api.types.is_integer_dtype(df["word_count"]),
        "Column 'word_count' has integer dtype",
        f"Actual: {df['word_count'].dtype}",
    )
    v.check(
        pd.api.types.is_bool_dtype(df["is_empty"]),
        "Column 'is_empty' has boolean dtype",
        f"Actual: {df['is_empty'].dtype}",
    )

    # 3. SỐ LƯỢNG BẢN GHI VÀ TÍNH DUY NHẤT CỦA ID
    v.log("\n[3] Document ID & Row Count")
    v.check(
        len(df) == 170824,
        "Row count matches raw content total (170,824)",
        f"Actual: {len(df):,}",
    )
    v.check(df["id"].notna().all(), "No missing ID values")
    v.check(df["id"].is_unique, "All Document IDs are unique")

    # 4. CHẤT LƯỢNG NỘI DUNG VÀ TÍNH ĐỒNG BỘ CỦA CỜ IS_EMPTY
    v.log("\n[4] Text extraction & is_empty consistency")
    expected_empty = df["char_count"] == 0
    empty_mismatch = int((df["is_empty"] != expected_empty).sum())
    v.check(
        empty_mismatch == 0,
        "is_empty flag strictly matches char_count == 0",
        f"Mismatches: {empty_mismatch}",
    )

    empty_count = int(df["is_empty"].sum())
    non_empty_count = len(df) - empty_count
    v.log(
        f"• Số văn bản có nội dung (retrievable): {non_empty_count:,} "
        f"({non_empty_count / len(df) * 100:.2f}%)"
    )
    v.log(
        f"• Số văn bản rỗng nội dung (chỉ có metadata): {empty_count:,} "
        f"({empty_count / len(df) * 100:.2f}%)"
    )

    # Kiểm tra chuẩn hóa trên tập mẫu non-empty
    non_empty_df = df[~df["is_empty"]]
    sample_size = min(5000, len(non_empty_df))
    if sample_size > 0:
        sample_texts = non_empty_df["text"].sample(n=sample_size, random_state=42)

        non_nfc = sample_texts.apply(
            lambda t: unicodedata.normalize("NFC", str(t)) != str(t)
        ).sum()
        has_nbsp = sample_texts.apply(lambda t: "\xa0" in str(t)).sum()
        has_cr = sample_texts.apply(lambda t: "\r" in str(t)).sum()

        v.check(
            non_nfc == 0,
            f"Sampled texts ({sample_size:,}) are 100% NFC normalized",
            f"Non-NFC count: {non_nfc}",
        )
        v.check(
            has_nbsp == 0,
            f"Sampled texts ({sample_size:,}) contain no NBSP",
            f"NBSP count: {has_nbsp}",
        )
        v.check(
            has_cr == 0,
            f"Sampled texts ({sample_size:,}) contain no CR/CRLF",
            f"CR count: {has_cr}",
        )

        # 5. MỨC ĐỘ SẠCH CỦA TÀN DƯ HTML
        v.log("\n[5] HTML artifact cleanliness")
        strict_html_pattern = (
            r"<\s*/?(?:div|span|html|body|script|style)\b|(?<!\w)</?p[>\s]"
        )
        html_residue = int(
            sample_texts.str.contains(strict_html_pattern, case=False, regex=True).sum()
        )
        v.check(
            html_residue == 0,
            "No standard HTML residue tags found in sample",
            f"Residual count: {html_residue}",
        )
    else:
        v.log("\n[5] HTML artifact cleanliness: Skipped (no non-empty texts to sample)")

    # 6. ĐỘ BAO PHỦ CỦA METADATA (Lọc chuỗi rỗng thực tế)
    v.log("\n[6] Metadata integration coverage")
    valid_titles = df["title"].notna() & (df["title"].astype(str).str.strip() != "")
    valid_types = df["loai_van_ban"].notna() & (
        df["loai_van_ban"].astype(str).str.strip() != ""
    )

    title_coverage = int(valid_titles.sum())
    type_coverage = int(valid_types.sum())

    v.log(f"• Valid title coverage    : {title_coverage / len(df) * 100:.2f}%")
    v.log(f"• Valid doc type coverage : {type_coverage / len(df) * 100:.2f}%")
    v.check(
        title_coverage > 0.95 * len(df),
        "Metadata valid title coverage > 95%",
        f"Actual: {title_coverage / len(df) * 100:.2f}%",
    )

    # TỔNG KẾT
    v.log("\n" + "=" * 80)
    if not v.has_failed:
        v.log("RESULT: LEGAL DOCUMENTS VALIDATION PASSED")
        verdict = 0
    else:
        v.log("RESULT: LEGAL DOCUMENTS VALIDATION FAILED")
        verdict = 1
    v.log("=" * 80)

    v.save_report()
    v.log(f"Báo cáo kiểm định đã được xuất tại: {report_file}")
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kiểm định artifact legal_documents_preprocessed.parquet."
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
    exit_code = validate_legal_documents(input_file=args.input, report_file=args.report)
    sys.exit(exit_code)