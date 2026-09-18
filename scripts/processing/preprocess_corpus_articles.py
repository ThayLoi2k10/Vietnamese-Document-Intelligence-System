# scripts/processing/preprocess_corpus_articles.py
"""Pipeline tiền xử lý văn bản cho Legal Retrieval Corpus.

Thực hiện:
    - Chuẩn hóa Unicode NFC
    - Loại bỏ NBSP (\u00a0)
    - Chuẩn hóa xuống dòng (\r\n -> \n)
    - Gộp khoảng trắng ngang nhưng bảo tồn \n và \t
    - Tạo các cột metadata phái sinh và khóa schema 10 cột

Input:  data/processed/legal_retrieval/corpus_articles.parquet
Output: data/processed/legal_retrieval/corpus_articles_preprocessed.parquet
"""

import argparse
from pathlib import Path
import re
import sys
from typing import Any
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
INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_retrieval"
    / "corpus_articles.parquet"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "legal_retrieval"
OUTPUT_FILE = OUTPUT_DIR / "corpus_articles_preprocessed.parquet"

SCHEMA_COLS = [
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


def apply_nfc(text: str) -> str:
    """Chuẩn hóa chuỗi về dạng Unicode NFC."""
    return unicodedata.normalize("NFC", text)


def apply_nbsp_to_space(text: str) -> str:
    """Chuyển đổi ký tự non-breaking space thành space chuẩn."""
    return text.replace("\u00a0", " ")


def remove_carriage_returns(text: str) -> str:
    """Đồng bộ các dạng ngắt dòng về ký tự newline chuẩn \\n."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def collapse_spaces_preserve_tabs(text: str) -> str:
    """Thu gọn khoảng trắng ngang liên tiếp thành 1 space.
    
    BẢO TOÀN ký tự xuống dòng \\n VÀ ký tự thụt lề \\t.
    """
    lines = text.split("\n")
    cleaned_lines = [
        re.sub(r"[^\S\r\n\t]+", " ", line).strip(" ") for line in lines
    ]
    return "\n".join(cleaned_lines)


def clean_legal_text(text: str) -> str:
    """Áp dụng toàn bộ quy trình làm sạch văn bản pháp luật."""
    if not isinstance(text, str):
        return ""
    t = apply_nfc(text)
    t = remove_carriage_returns(t)
    t = apply_nbsp_to_space(t)
    t = collapse_spaces_preserve_tabs(t)
    return t


def is_semantically_empty(text: Any) -> bool:
    """Kiểm tra giá trị rỗng hoặc chỉ chứa khoảng trắng."""
    if pd.isna(text):
        return True
    return str(text).strip() == ""


def preprocess_corpus_articles(force: bool = False) -> None:
    """Thực thi pipeline tiền xử lý cho Legal Retrieval Corpus."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file đầu vào: {INPUT_FILE}. "
            "Hãy chạy scripts/processing/build_corpus_article.py trước."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not force and OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:
        file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
        print(
            f"Artifact đã tồn tại: {OUTPUT_FILE} ({file_size_mb:.2f} MB). "
            "Bỏ qua xử lý (Dùng --force để tạo lại)."
        )
        return

    print("=" * 80)
    print("PIPELINE: PREPROCESSING LEGAL RETRIEVAL CORPUS")
    print("=" * 80)
    print(f"Đọc dữ liệu từ: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    print(f"Tổng số bản ghi đầu vào: {len(df):,}")

    # 1. Lưu text thô ban đầu vào text_raw
    df["text_raw"] = df["text"].fillna("").astype(str)

    # 2. Tiền xử lý text bảo toàn cấu trúc
    print("Đang chuẩn hóa (Unicode NFC, loại NBSP, chuẩn hóa Newline, giữ TAB)...")
    df["text_preprocessed"] = df["text_raw"].apply(clean_legal_text)

    # 3. Cột text chính thức nhận text đã làm sạch
    df["text"] = df["text_preprocessed"]

    # 4. Tính toán metadata phái sinh
    print("Tính toán metadata phái sinh...")
    df["char_count_preprocessed"] = df["text_preprocessed"].str.len()
    df["word_count_preprocessed"] = (
        df["text_preprocessed"].str.split().str.len()
    )
    df["is_empty_preprocessed"] = df["text_preprocessed"].apply(
        is_semantically_empty
    )

    # 5. Khóa đúng cấu trúc 10 cột theo quy chuẩn
    df = df[SCHEMA_COLS]

    # 6. Ghi file Parquet (atomic write)
    print(f"Lưu artifact tại: {OUTPUT_FILE}")
    temp_file = OUTPUT_FILE.with_suffix(OUTPUT_FILE.suffix + ".tmp")
    df.to_parquet(temp_file, engine="pyarrow", index=False)
    temp_file.replace(OUTPUT_FILE)
    print(f"-> Hoàn tất Preprocessing! Shape: {df.shape}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tiền xử lý văn bản cho corpus_articles.parquet."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Buộc chạy lại pipeline ngay cả khi artifact đã tồn tại.",
    )
    args = parser.parse_args()
    preprocess_corpus_articles(force=args.force)