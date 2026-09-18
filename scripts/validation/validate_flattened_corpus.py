# scripts/validation/validate_flattened_corpus.py
"""Script kiểm định chất lượng artifact corpus_articles.parquet đối chiếu với legal_corpus.json.

Kiểm tra:
    1. Kiểm tra tồn tại dữ liệu nguồn và bảo vệ chống rỗng
    2. Kiểm định tính toàn vẹn của schema và dtype cốt lõi trước khi truy cập cột (chống KeyError/TypeError)
    3. Kiểm định tính duy nhất và liên tục của global_article_id (0 -> N-1)
    4. Tái tạo corpus phẳng từ legal_corpus.json thô để so sánh số lượng bản ghi
    5. So sánh chính xác từng trường (doc_id, law_id, aid, text)
    6. Kiểm định tính nhất quán của cờ is_empty và các trường phái sinh (char_count, word_count)
"""

import argparse
import json
from pathlib import Path
import sys
from typing import List, Set
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
DEFAULT_CORPUS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_retrieval"
    / "corpus_articles.parquet"
)
DEFAULT_RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "legal_retrieval"
    / "legal_corpus.json"
)

EXPECTED_COLUMNS: Set[str] = {
    "global_article_id",
    "doc_id",
    "law_id",
    "aid",
    "text",
    "char_count",
    "word_count",
    "is_empty",
}


class FlattenedCorpusValidator:
    """Quản lý quá trình kiểm định tính toàn vẹn của bảng corpus articles."""

    def __init__(self) -> None:
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


def validate_flattened_corpus(corpus_path: Path, raw_path: Path) -> int:
    """Thực thi toàn bộ quy trình kiểm định corpus_articles.parquet."""
    if not corpus_path.exists():
        print(f"Lỗi: Không tìm thấy file corpus đã xử lý: {corpus_path}")
        return 1
    if not raw_path.exists():
        print(f"Lỗi: Không tìm thấy file JSON thô: {raw_path}")
        return 1

    v = FlattenedCorpusValidator()
    v.log("=" * 80)
    v.log("VALIDATION: FLATTENED LEGAL RETRIEVAL CORPUS")
    v.log("=" * 80)

    # 1. NẠP DỮ LIỆU & KIỂM TRA RỖNG
    v.log("\n[1] Loading data...")
    df = pd.read_parquet(corpus_path)

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_corpus = json.load(f)

    v.log(f"Processed corpus : {len(df):,} rows")
    v.log(f"Raw documents    : {len(raw_corpus):,} docs")

    if not v.check(len(df) > 0, "Processed corpus is not empty", f"Actual rows: {len(df)}"):
        v.log("\n" + "=" * 80)
        v.log("RESULT: FLATTENED CORPUS VALIDATION FAILED (Empty Processed Corpus)")
        v.log("=" * 80)
        return 1

    # 2. KIỂM ĐỊNH SCHEMA VÀ DTYPE CỐT LÕI
    v.log("\n[2] Schema and Data Types validation")
    actual_columns = set(df.columns)
    missing = EXPECTED_COLUMNS - actual_columns
    schema_ok = len(missing) == 0
    v.check(schema_ok, "Required columns exist", f"Missing: {missing}")

    if not schema_ok:
        v.log("\n" + "=" * 80)
        v.log("RESULT: FLATTENED CORPUS VALIDATION FAILED (Missing Columns)")
        v.log("=" * 80)
        return 1

    v.check(
        pd.api.types.is_integer_dtype(df["global_article_id"]),
        "Column 'global_article_id' has integer dtype",
        f"Actual: {df['global_article_id'].dtype}",
    )
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

    # 3. GLOBAL ARTICLE ID
    v.log("\n[3] Global article ID validation")
    ids = df["global_article_id"]
    v.check(ids.notna().all(), "No missing global_article_id")
    v.check(ids.is_unique, "global_article_id is unique")

    is_continuous = (ids.values == range(len(df))).all()
    v.check(
        is_continuous,
        "global_article_id is strictly continuous from 0 to len-1",
        f"Actual range: {ids.min()} -> {ids.max():,}",
    )

    # 4. TÁI TẠO BẢNG ĐỐI CHIẾU TỪ RAW JSON
    v.log("\n[4] Reconstructing expected flattened corpus")
    expected_rows = []
    for doc in raw_corpus:
        doc_id = doc.get("id")
        law_id = str(doc.get("law_id", "")).strip()

        for article in doc.get("content", []):
            expected_rows.append(
                {
                    "doc_id": doc_id,
                    "law_id": law_id,
                    "aid": article.get("aid"),
                    "text": article.get("content_Article", ""),
                }
            )

    expected_df = pd.DataFrame(expected_rows)
    v.log(f"Expected article count: {len(expected_df):,}")

    count_match = len(df) == len(expected_df)
    v.check(
        count_match,
        "Article count matches raw corpus exactly",
        f"Processed={len(df):,}, Expected={len(expected_df):,}",
    )

    if not count_match:
        v.log("\n" + "=" * 80)
        v.log("RESULT: FLATTENED CORPUS VALIDATION FAILED (Row Count Mismatch)")
        v.log("=" * 80)
        return 1

    # 5. SO SÁNH CHÍNH XÁC TỪNG TRƯỜNG DỮ LIỆU
    v.log("\n[5] Field integrity validation")
    fields = ["doc_id", "law_id", "aid", "text"]

    for field in fields:
        processed_series = df[field].fillna("").astype(str).reset_index(drop=True)
        expected_series = expected_df[field].fillna("").astype(str).reset_index(drop=True)

        mismatch_mask = processed_series != expected_series
        mismatch_count = int(mismatch_mask.sum())

        v.check(
            mismatch_count == 0,
            f"Field '{field}' preserved correctly",
            f"Mismatches: {mismatch_count:,}",
        )

    # 6. KIỂM TRA ĐIỀU LUẬT RỖNG (IS_EMPTY)
    v.log("\n[6] Empty article validation")
    expected_empty = (
        expected_df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )
    actual_empty = df["is_empty"].fillna(False).astype(bool)

    empty_mismatches = int((actual_empty != expected_empty).sum())
    v.check(
        empty_mismatches == 0,
        "is_empty flag strictly matches raw text emptiness",
        f"Expected empty: {int(expected_empty.sum()):,}, Actual empty: {int(actual_empty.sum()):,}",
    )

    # 7. KIỂM TRA CÁC TRƯỜNG THỐNG KÊ PHÁI SINH
    v.log("\n[7] Derived field validation")
    stripped_text = expected_df["text"].fillna("").astype(str).str.strip()
    expected_char_count = stripped_text.str.len()
    expected_word_count = stripped_text.str.split().str.len()

    char_mismatches = int((df["char_count"] != expected_char_count).sum())
    word_mismatches = int((df["word_count"] != expected_word_count).sum())

    v.check(
        char_mismatches == 0,
        "char_count matches recalculated ground truth",
        f"Mismatches: {char_mismatches:,}",
    )
    v.check(
        word_mismatches == 0,
        "word_count matches recalculated ground truth",
        f"Mismatches: {word_mismatches:,}",
    )

    # 8. KẾT QUẢ TỔNG HỢP
    v.log("\n" + "=" * 80)
    if not v.has_failed:
        v.log("RESULT: FLATTENED CORPUS VALIDATION PASSED")
        verdict = 0
    else:
        v.log("RESULT: FLATTENED CORPUS VALIDATION FAILED")
        verdict = 1
    v.log("=" * 80)

    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kiểm định tính toàn vẹn của corpus_articles.parquet so với legal_corpus.json."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_FILE,
        help="Đường dẫn tới file corpus_articles.parquet.",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=DEFAULT_RAW_FILE,
        help="Đường dẫn tới file legal_corpus.json.",
    )
    args = parser.parse_args()
    exit_code = validate_flattened_corpus(corpus_path=args.corpus, raw_path=args.raw)
    sys.exit(exit_code)