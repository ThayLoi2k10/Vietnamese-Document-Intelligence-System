# scripts/benchmarks/analyze_qa_leakage_and_distribution.py
"""Script kiểm chứng QA source, phân tích phân phối taxonomy và rủi ro rò rỉ ngữ cảnh (Content Leakage).

Thực hiện:
    1. Kiểm tra tồn tại dữ liệu và schema tối thiểu của file QA Parquet
    2. Phân tích phân phối ma trận chéo Taxonomy (difficulty x question_type)
    3. Thống kê phân vị số câu hỏi trên mỗi document ID
    4. Chuẩn hóa văn bản và băm MD5 ngữ cảnh (article_content) để phát hiện trùng lặp
    5. Cảnh báo kiến trúc nếu tồn tại nhiều doc_id chia sẻ chung nội dung điều luật
"""

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any, Set
import unicodedata
import numpy as np
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
DEFAULT_QA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "legal_qa" / "legal_qa_preprocessed.parquet"
)

REQUIRED_COLUMNS: Set[str] = {
    "doc_id",
    "article_content",
    "difficulty",
    "question_type",
}


def clean_and_hash_content(text: Any) -> str:
    """Chuẩn hóa Unicode NFC, gộp khoảng trắng thừa và băm MD5 ngữ cảnh."""
    if not isinstance(text, str) or not text.strip():
        return ""
    normalized = unicodedata.normalize("NFC", text)
    clean_text = " ".join(normalized.split())
    return hashlib.md5(clean_text.encode("utf-8")).hexdigest()


def analyze_qa_leakage_and_distribution(qa_path: Path) -> int:
    """Thực thi phân tích phân phối và rủi ro rò rỉ ngữ cảnh."""
    print("=" * 60)
    print("KIỂM CHỨNG QA SOURCE & PHÂN TÍCH NGUY CƠ LEAKAGE")
    print("=" * 60)

    if not qa_path.exists():
        print(f"[FAIL] Không tìm thấy file dữ liệu: {qa_path}")
        return 1

    print(f"Đang đọc dữ liệu QA: {qa_path.name}...")
    try:
        df = pq.read_table(qa_path).to_pandas()
    except Exception as e:
        print(f"[FAIL] Lỗi khi đọc file Parquet: {e}")
        return 1

    total_records = len(df)
    print(f"Tổng số bản ghi QA: {total_records:,}")

    if total_records == 0:
        print("[FAIL] Tập dữ liệu QA rỗng bản ghi.")
        return 1

    # Kiểm tra schema tối thiểu
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        print(f"[FAIL] Thiếu các cột bắt buộc: {missing_cols}")
        return 1

    # 1. Phân tích ma trận chéo Taxonomy (difficulty x question_type)
    print("\n" + "-" * 20 + " 1. PHÂN PHỐI TAXONOMY " + "-" * 20)
    cross_tab = pd.crosstab(
        df["difficulty"],
        df["question_type"],
        margins=True,
        margins_name="Total",
    )
    cross_tab_pct = (
        pd.crosstab(
            df["difficulty"],
            df["question_type"],
            normalize=True,
            margins=True,
            margins_name="Total",
        )
        * 100
    )

    print("Bảng đếm số lượng (Counts):")
    print(cross_tab)
    print("\nBảng tỷ lệ phần trăm (%):")
    print(cross_tab_pct.round(2))

    # 2. Phân tích phân bố số câu hỏi trên mỗi doc_id
    print("\n" + "-" * 20 + " 2. PHÂN BỐ CÂU HỎI TRÊN MỖI DOC_ID " + "-" * 20)
    qa_per_doc = df.groupby("doc_id").size()
    total_docs = len(qa_per_doc)
    print(f"Tổng số unique doc_id: {total_docs:,}")
    print(f"  - Min câu hỏi / doc: {qa_per_doc.min()}")
    print(f"  - 25th percentile: {np.percentile(qa_per_doc, 25)}")
    print(f"  - Median: {qa_per_doc.median()}")
    print(f"  - Mean: {qa_per_doc.mean():.2f}")
    print(f"  - 75th percentile: {np.percentile(qa_per_doc, 75)}")
    print(f"  - Max câu hỏi / doc: {qa_per_doc.max()}")

    freq_dist = qa_per_doc.value_counts().sort_index()
    print("\nTần suất số lượng câu hỏi trên mỗi doc_id:")
    for num_qa, count in freq_dist.items():
        print(
            f"  - {num_qa} QA/doc: {count:,} docs ({count / total_docs * 100:.2f}%)"
        )

    # 3. Phân tích rủi ro rò rỉ ngữ cảnh (Content Leakage) sau khi chuẩn hóa
    print("\n" + "-" * 20 + " 3. ĐIỀU TRA CONTENT LEAKAGE (CHUẨN HÓA NFC) " + "-" * 20)
    df["content_hash"] = df["article_content"].apply(clean_and_hash_content)

    empty_hashes = (df["content_hash"] == "").sum()
    if empty_hashes > 0:
        print(f"[WARN] Có {empty_hashes:,} bản ghi có article_content rỗng.")

    unique_hashes = df[df["content_hash"] != ""]["content_hash"].nunique()
    print(f"Số lượng unique article_content (theo normalized hash): {unique_hashes:,}")

    # Kiểm tra xem có bao nhiêu hash xuất hiện ở nhiều doc_id khác nhau
    valid_df = df[df["content_hash"] != ""]
    hash_to_docs = valid_df.groupby("content_hash")["doc_id"].nunique()
    shared_hashes = hash_to_docs[hash_to_docs > 1]
    num_shared_hashes = len(shared_hashes)

    print(
        f"Số lượng nội dung (hash) bị dùng chung bởi >= 2 doc_id: {num_shared_hashes:,}"
    )

    if num_shared_hashes > 0:
        affected_records = valid_df[valid_df["content_hash"].isin(shared_hashes.index)]
        affected_docs = affected_records["doc_id"].nunique()
        print(f"  - Số doc_id bị ảnh hưởng: {affected_docs:,}")
        print(f"  - Số cặp QA bị ảnh hưởng: {len(affected_records):,}")
        print(
            "\n[KẾT LUẬN KIẾN TRÚC]: Tồn tại doc_id khác nhau có chung article_content."
        )
        print(
            "  -> BẮT BUỘC: Sử dụng StratifiedGroupKFold theo content_hash (không dùng doc_id)"
        )
        print("     để bảo đảm tuyệt đối 0% data leakage giữa Train, Val và Test.")
    else:
        print(
            "[XÁC NHẬN]: Không phát hiện normalized article_content "
            "được chia sẻ giữa các doc_id khác nhau."
        )

    print("\n" + "=" * 60)
    print("HOÀN THÀNH PHÂN TÍCH QA LEAKAGE & DISTRIBUTION.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phân tích phân phối QA taxonomy và nguy cơ rò rỉ ngữ cảnh."
    )
    parser.add_argument(
        "--qa-path",
        type=Path,
        default=DEFAULT_QA_PATH,
        help="Đường dẫn file legal_qa_preprocessed.parquet.",
    )
    args = parser.parse_args()

    exit_code = analyze_qa_leakage_and_distribution(qa_path=args.qa_path)
    sys.exit(exit_code)