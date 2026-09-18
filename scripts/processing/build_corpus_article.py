# scripts/processing/build_corpus_article.py
"""Script làm phẳng legal_corpus.json thành bảng điều khoản corpus_articles.parquet.

Input:  data/raw/legal_retrieval/legal_corpus.json
Output: data/processed/legal_retrieval/corpus_articles.parquet
Bảo toàn Global Article ID từ 0 đến 59,635.
"""

import argparse
import json
from pathlib import Path
import sys
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
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "legal_retrieval" / "legal_corpus.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "legal_retrieval"
OUTPUT_FILE = OUTPUT_DIR / "corpus_articles.parquet"


def build_corpus_articles(force: bool = False) -> None:
    """Làm phẳng cấu trúc phân cấp văn bản pháp luật thành từng điều khoản đơn lẻ."""
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file nguồn: {RAW_FILE}. "
            "Hãy đảm bảo dataset legal_retrieval đã được đặt đúng trong data/raw/."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not force and OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:
        file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
        print(
            f"Artifact đã tồn tại: {OUTPUT_FILE} ({file_size_mb:.2f} MB). "
            "Bỏ qua xử lý (Dùng --force để tạo lại)."
        )
        return

    print(f"Đang nạp file dữ liệu thô: {RAW_FILE}")
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        raw_corpus = json.load(f)

    flat_articles = []
    global_idx = 0

    # Duyệt tuần tự để bảo toàn tuyệt đối Global Index khớp với ground truth
    for doc in raw_corpus:
        doc_id = doc.get("id")
        law_id = str(doc.get("law_id", "")).strip()
        articles = doc.get("content", [])

        for item in articles:
            aid = item.get("aid")
            raw_text = item.get("content_Article", "")
            clean_text = raw_text.strip() if raw_text else ""

            flat_articles.append(
                {
                    "global_article_id": global_idx,
                    "doc_id": doc_id,
                    "law_id": law_id,
                    "aid": aid,
                    "text": raw_text,
                    "char_count": len(clean_text),
                    "word_count": len(clean_text.split()),
                    "is_empty": len(clean_text) == 0,
                }
            )
            global_idx += 1

    df_corpus = pd.DataFrame(flat_articles)
    temp_file = OUTPUT_FILE.with_suffix(OUTPUT_FILE.suffix + ".tmp")
    df_corpus.to_parquet(temp_file, index=False)
    temp_file.replace(OUTPUT_FILE)

    print("=" * 80)
    print("KẾT QUẢ FLATTEN LEGAL CORPUS")
    print("=" * 80)
    print(f"- Tổng số Điều luật (Rows):        {len(df_corpus):,}")
    print(f"- Số văn bản luật cha (Docs):      {df_corpus['doc_id'].nunique():,}")
    print(
        f"- Global Index chạy từ:            {df_corpus['global_article_id'].min()} -> "
        f"{df_corpus['global_article_id'].max():,}"
    )
    print(f"- Số Điều luật rỗng (is_empty):   {df_corpus['is_empty'].sum()} Điều")
    print(f"-> Đã lưu bảng phẳng tại:          {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Làm phẳng legal_corpus.json thành corpus_articles.parquet."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Buộc tạo lại file ngay cả khi artifact đã tồn tại.",
    )
    args = parser.parse_args()
    build_corpus_articles(force=args.force)