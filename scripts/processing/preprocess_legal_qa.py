# scripts/processing/preprocess_legal_qa.py
"""Pipeline tiền xử lý và làm phẳng tập dữ liệu Vietnamese Legal QA.

Thực hiện:
    - Làm phẳng mảng generated_qa_pairs từ cấp document xuống cấp từng cặp QA
    - Kiểm tra cấu trúc chặt chẽ (list/tuple và dict cho từng cặp QA)
    - Chuẩn hóa văn bản câu hỏi, câu trả lời và nội dung văn bản liên quan
    - Chuẩn hóa phân loại taxonomy (difficulty, question_type) về dạng chuẩn
    - Xuất artifact Parquet theo cơ chế ghi nguyên tử (atomic write)

Input:  data/raw/legal_qa/vietnamese_legal_qa.parquet
Output: data/processed/legal_qa/legal_qa_preprocessed.parquet
"""

import argparse
from pathlib import Path
import re
import sys
from typing import Any, Dict
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
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "legal_qa" / "vietnamese_legal_qa.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "legal_qa"
OUTPUT_FILE = OUTPUT_DIR / "legal_qa_preprocessed.parquet"
TEMP_FILE = OUTPUT_DIR / "legal_qa_preprocessed.parquet.tmp"

DIFFICULTY_MAP = {
    "easy": "easy",
    "dễ": "easy",
    "medium": "medium",
    "trung bình": "medium",
    "hard": "hard",
    "khó": "hard",
}

QUESTION_TYPE_MAP = {
    "factual": "factual",
    "thực tế": "factual",
    "interpretation": "interpretation",
    "giải thích": "interpretation",
    "application": "application",
    "ứng dụng": "application",
    "analytical": "analytical",
}

FINAL_COLS = [
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


def clean_vietnamese_text(text: Any) -> str:
    """Chuẩn hóa văn bản tiếng Việt bảo tồn ngắt dòng."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFC", text)
    t = t.replace("\u00a0", " ")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    lines = t.split("\n")
    cleaned_lines = [
        re.sub(r"[^\S\r\n\t]+", " ", line).strip(" ") for line in lines
    ]
    return "\n".join(cleaned_lines).strip()


def normalize_taxonomy(val: Any, mapping: Dict[str, str], default: str = "unknown") -> str:
    """Chuẩn hóa giá trị phân loại về nhãn quy chuẩn."""
    if not val or pd.isna(val):
        return default
    normalized_val = unicodedata.normalize("NFC", str(val)).strip().lower()
    return mapping.get(normalized_val, normalized_val)


def preprocess_legal_qa(force: bool = False) -> None:
    """Thực thi pipeline tiền xử lý cho Vietnamese Legal QA."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file nguồn: {INPUT_FILE}. "
            "Hãy chạy scripts/data_acquisition/download_legal_qa.py trước."
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
    print("PIPELINE: PREPROCESSING VIETNAMESE LEGAL QA")
    print("=" * 80)
    print(f"Đọc dữ liệu từ: {INPUT_FILE}")
    df_raw = pd.read_parquet(INPUT_FILE)
    print(f"Số văn bản thô đầu vào: {len(df_raw):,}")

    if len(df_raw) == 0:
        raise ValueError("Tập dữ liệu đầu vào rỗng, không thể xử lý.")

    flat_records = []
    qa_counter = 0

    print("Đang bóc tách (flatten) mảng QA, chuẩn hóa text và ánh xạ taxonomy...")
    for doc_idx, row in enumerate(df_raw.to_dict(orient="records")):
        doc_name = str(row.get("doc_name", "")).strip()
        doc_type = str(row.get("doc_type_name", "")).strip()
        cleaned_content = clean_vietnamese_text(row.get("article_content", ""))
        gen_time = float(row.get("generation_time", 0.0) or 0.0)

        qa_pairs = row.get("generated_qa_pairs")
        if qa_pairs is None:
            continue
        if not isinstance(qa_pairs, (list, tuple)):
            raise TypeError(
                f"generated_qa_pairs tại doc_id={doc_idx} phải là list/tuple, "
                f"nhưng nhận được {type(qa_pairs).__name__}"
            )

        for qa_idx, qa in enumerate(qa_pairs):
            if not isinstance(qa, dict):
                raise TypeError(
                    f"QA không hợp lệ tại doc_id={doc_idx}, "
                    f"qa_index={qa_idx}: expected dict, "
                    f"got {type(qa).__name__}"
                )

            raw_diff = qa.get("difficulty", "")
            raw_type = qa.get("question_type", "")

            flat_records.append(
                {
                    "qa_id": qa_counter,
                    "doc_id": doc_idx,
                    "doc_name": doc_name,
                    "doc_type_name": doc_type,
                    "article_content": cleaned_content,
                    "question": clean_vietnamese_text(qa.get("question", "")),
                    "answer": clean_vietnamese_text(qa.get("answer", "")),
                    "difficulty": normalize_taxonomy(raw_diff, DIFFICULTY_MAP),
                    "question_type": normalize_taxonomy(raw_type, QUESTION_TYPE_MAP),
                    "generation_time": gen_time,
                }
            )
            qa_counter += 1

    df_flat = pd.DataFrame(flat_records)
    print(f"Tổng số cặp QA sau khi làm phẳng: {len(df_flat):,}")

    df_flat = df_flat[FINAL_COLS]

    if TEMP_FILE.exists():
        TEMP_FILE.unlink()

    try:
        df_flat.to_parquet(TEMP_FILE, engine="pyarrow", index=False)
        TEMP_FILE.replace(OUTPUT_FILE)
    finally:
        if TEMP_FILE.exists():
            TEMP_FILE.unlink(missing_ok=True)

    print(f"Đã lưu artifact tại: {OUTPUT_FILE}")
    print(f"Shape: {df_flat.shape}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Làm phẳng và chuẩn hóa Vietnamese Legal QA thành file Parquet."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Buộc xử lý lại toàn bộ ngay cả khi artifact đã tồn tại.",
    )
    args = parser.parse_args()
    preprocess_legal_qa(force=args.force)