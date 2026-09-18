# scripts/validation/validate_preprocessed_corpus.py
"""Script kiểm định chất lượng artifact corpus_articles_preprocessed.parquet.

Kiểm tra:
    1. Kiểm tra tồn tại dữ liệu và bảo vệ chống rỗng
    2. Kiểm định tính toàn vẹn của schema 10 cột và kiểu dữ liệu (Data Types)
    3. Bảo toàn số lượng bản ghi và tính liên tục của global_article_id
    4. Bảo toàn các trường phi văn bản (doc_id, law_id, aid)
    5. Bảo toàn nguyên vẹn text_raw so với text ban đầu
    6. Đồng bộ 100% giữa cột text chính thức và text_preprocessed
    7. Chuẩn hóa Unicode NFC, sạch ký tự NBSP, CR/CRLF và bảo toàn ký tự thụt lề TAB
    8. Tính nhất quán của cờ is_empty_preprocessed và các trường phái sinh
"""

import argparse
from pathlib import Path
import sys
from typing import Any, List
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
DEFAULT_PROCESSED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_retrieval"
    / "corpus_articles_preprocessed.parquet"
)
DEFAULT_RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_retrieval"
    / "corpus_articles.parquet"
)
DEFAULT_REPORT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_retrieval"
    / "validation_report.txt"
)

REQUIRED_COLUMNS = [
    "global_article_id",
    "doc_id",
    "law_id",
    "aid",
    "text",
    "text_raw",
    "text_preprocessed",
    "char_count_preprocessed",
    "word_count_preprocessed",
    "is_empty_preprocessed",
]


def is_semantically_empty(text: Any) -> bool:
    """Kiểm tra giá trị rỗng hoặc chỉ chứa khoảng trắng."""
    if pd.isna(text):
        return True
    return str(text).strip() == ""


class ValidationRunner:
    """Quản lý quá trình kiểm định và ghi nhận báo cáo."""

    def __init__(self, report_path: Path) -> None:
        self.report_path = report_path
        self.logs: List[str] = []
        self.has_failed: bool = False

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


def validate_preprocessed_corpus(
    processed_file: Path, raw_file: Path, report_file: Path
) -> int:
    """Thực thi toàn bộ quy trình kiểm định corpus_articles_preprocessed.parquet."""
    if not processed_file.exists():
        print(f"Lỗi: Không tìm thấy file processed: {processed_file}")
        return 1
    if not raw_file.exists():
        print(f"Lỗi: Không tìm thấy file raw articles: {raw_file}")
        return 1

    runner = ValidationRunner(report_file)

    runner.log("=" * 80)
    runner.log("VALIDATION: PREPROCESSED LEGAL RETRIEVAL CORPUS")
    runner.log("=" * 80)

    # 1. LOAD DATA & GUARD RỖNG
    runner.log("\n[1] Loading data...")
    df = pd.read_parquet(processed_file)
    raw_df = pd.read_parquet(raw_file)
    runner.log(f"Preprocessed rows : {len(df):,}")
    runner.log(f"Raw rows          : {len(raw_df):,}")

    if not runner.check(len(df) > 0, "Processed dataset is not empty", f"Actual rows: {len(df)}"):
        runner.log("\n" + "=" * 80)
        runner.log("RESULT: PREPROCESSED CORPUS VALIDATION FAILED (Empty Dataset)")
        runner.log("=" * 80)
        runner.save_report()
        return 1

    # 2. SCHEMA & DATA TYPES VALIDATION (Guard KeyError)
    runner.log("\n[2] Schema & Data Types validation")
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    schema_ok = len(missing) == 0
    runner.check(schema_ok, "Required columns exist", f"Missing: {missing}")

    if not schema_ok:
        runner.log("\n" + "=" * 80)
        runner.log("RESULT: PREPROCESSED CORPUS VALIDATION FAILED (Missing Required Columns)")
        runner.log("=" * 80)
        runner.save_report()
        return 1

    runner.check(
        pd.api.types.is_integer_dtype(df["global_article_id"]),
        "Column 'global_article_id' has integer dtype",
        f"Actual: {df['global_article_id'].dtype}",
    )
    runner.check(
        pd.api.types.is_integer_dtype(df["char_count_preprocessed"]),
        "Column 'char_count_preprocessed' has integer dtype",
        f"Actual: {df['char_count_preprocessed'].dtype}",
    )
    runner.check(
        pd.api.types.is_integer_dtype(df["word_count_preprocessed"]),
        "Column 'word_count_preprocessed' has integer dtype",
        f"Actual: {df['word_count_preprocessed'].dtype}",
    )
    runner.check(
        pd.api.types.is_bool_dtype(df["is_empty_preprocessed"]),
        "Column 'is_empty_preprocessed' has boolean dtype",
        f"Actual: {df['is_empty_preprocessed'].dtype}",
    )

    # 3. ROW & ID INTEGRITY
    runner.log("\n[3] Row and ID integrity")
    runner.check(
        len(df) == len(raw_df),
        "Row count preserved",
        f"Raw={len(raw_df):,}, Processed={len(df):,}",
    )
    runner.check(df["global_article_id"].notna().all(), "No missing global_article_id")
    runner.check(df["global_article_id"].is_unique, "global_article_id remains unique")

    is_continuous = (df["global_article_id"].values == range(len(df))).all()
    runner.check(
        is_continuous,
        "global_article_id remains strictly continuous",
        f"Range: {df['global_article_id'].min()} -> {df['global_article_id'].max():,}",
    )

    # 4. NON-TEXT FIELD PRESERVATION
    runner.log("\n[4] Non-text field preservation")
    for column in ["global_article_id", "doc_id", "law_id", "aid"]:
        mismatch = int((df[column].fillna("__NULL__") != raw_df[column].fillna("__NULL__")).sum())
        runner.check(
            mismatch == 0,
            f"{column} preserved",
            f"Mismatches: {mismatch:,}" if mismatch > 0 else "",
        )

    # 5. TEXT FIELD INTEGRITY & PRESERVATION
    runner.log("\n[5] Text fields integrity and preservation")
    raw_text_mismatch = int((df["text_raw"].fillna("__NULL__") != raw_df["text"].fillna("__NULL__")).sum())
    runner.check(
        raw_text_mismatch == 0,
        "text_raw exactly matches original text",
        f"Mismatches: {raw_text_mismatch:,}" if raw_text_mismatch > 0 else "",
    )

    # Kiểm tra cột text chính thức khớp với text_preprocessed
    official_text_mismatch = int((df["text"].fillna("") != df["text_preprocessed"].fillna("")).sum())
    runner.check(
        official_text_mismatch == 0,
        "Official 'text' column exactly matches 'text_preprocessed'",
        f"Mismatches: {official_text_mismatch:,}" if official_text_mismatch > 0 else "",
    )

    # 6. UNICODE NFC
    runner.log("\n[6] Unicode normalization")
    proc_texts = df["text_preprocessed"].fillna("").astype(str)
    non_nfc = int(proc_texts.apply(lambda t: unicodedata.normalize("NFC", t) != t).sum())
    runner.check(
        non_nfc == 0,
        "All text is NFC normalized",
        f"Non-NFC texts: {non_nfc:,}" if non_nfc > 0 else "",
    )

    # 7. NBSP
    runner.log("\n[7] NBSP validation")
    remaining_nbsp = int(proc_texts.apply(lambda t: "\xa0" in t).sum())
    runner.check(
        remaining_nbsp == 0,
        "No NBSP remains",
        f"Articles containing NBSP: {remaining_nbsp:,}" if remaining_nbsp > 0 else "",
    )

    # 8. NEWLINE VALIDATION
    runner.log("\n[8] Newline validation")
    crlf = int(proc_texts.apply(lambda t: "\r\n" in t).sum())
    cr = int(proc_texts.apply(lambda t: "\r" in t).sum())
    runner.check(
        crlf == 0 and cr == 0,
        "No CRLF or standalone CR remains",
        f"CRLF: {crlf:,}, CR: {cr:,}" if (crlf > 0 or cr > 0) else "",
    )

    # 9. STRUCTURAL TAB PRESERVATION
    runner.log("\n[9] Structural whitespace preservation")
    raw_tabs = int(raw_df["text"].fillna("").astype(str).str.count("\t").sum())
    proc_tabs = int(proc_texts.str.count("\t").sum())
    runner.log(f"Raw TAB characters        : {raw_tabs:,}")
    runner.log(f"Processed TAB characters  : {proc_tabs:,}")
    runner.check(
        raw_tabs == proc_tabs,
        "TAB characters preserved",
        f"Raw={raw_tabs:,}, Processed={proc_tabs:,}",
    )

    # 10. EMPTY ARTICLE VALIDATION
    runner.log("\n[10] Empty article validation")
    raw_empty = int(raw_df["text"].apply(is_semantically_empty).sum())
    proc_empty = int(proc_texts.apply(is_semantically_empty).sum())
    runner.log(f"Raw empty articles         : {raw_empty:,}")
    runner.log(f"Processed empty articles   : {proc_empty:,}")
    runner.check(
        raw_empty == proc_empty,
        "Empty article count preserved",
        f"Raw={raw_empty:,}, Processed={proc_empty:,}",
    )

    expected_empty_flag = proc_texts.apply(is_semantically_empty)
    flag_mismatch = int((df["is_empty_preprocessed"] != expected_empty_flag).sum())
    runner.check(
        flag_mismatch == 0,
        "is_empty_preprocessed matches semantic definition",
        f"Mismatches: {flag_mismatch:,}" if flag_mismatch > 0 else "",
    )

    # 11. DERIVED FIELD VALIDATION
    runner.log("\n[11] Derived field validation")
    expected_char = proc_texts.str.len()
    expected_word = proc_texts.str.split().str.len()

    char_mismatch = int((df["char_count_preprocessed"] != expected_char).sum())
    word_mismatch = int((df["word_count_preprocessed"] != expected_word).sum())
    runner.check(char_mismatch == 0, "char_count_preprocessed is correct", f"Mismatches: {char_mismatch}")
    runner.check(word_mismatch == 0, "word_count_preprocessed is correct", f"Mismatches: {word_mismatch}")

    # VERDICT & REPORT EXPORT
    runner.log("\n" + "=" * 80)
    if not runner.has_failed:
        runner.log("RESULT: PREPROCESSED CORPUS VALIDATION PASSED")
        verdict = 0
    else:
        runner.log("RESULT: PREPROCESSED CORPUS VALIDATION FAILED")
        verdict = 1
    runner.log("=" * 80)

    runner.save_report()
    runner.log(f"Báo cáo kiểm định đã được xuất tại: {report_file}")
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kiểm định artifact corpus_articles_preprocessed.parquet."
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=DEFAULT_PROCESSED_FILE,
        help="Đường dẫn tới file corpus_articles_preprocessed.parquet.",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=DEFAULT_RAW_FILE,
        help="Đường dẫn tới file corpus_articles.parquet gốc.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_FILE,
        help="Đường dẫn lưu file báo cáo kết quả kiểm định.",
    )
    args = parser.parse_args()
    exit_code = validate_preprocessed_corpus(
        processed_file=args.processed,
        raw_file=args.raw,
        report_file=args.report,
    )
    sys.exit(exit_code)