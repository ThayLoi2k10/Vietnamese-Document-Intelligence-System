# scripts/processing/preprocess_legal_documents.py
"""Pipeline tiền xử lý và kết hợp dữ liệu văn bản pháp luật (Legal Documents).

Thực hiện:
    - Làm sạch HTML từ content shards (12 partitions)
    - Chuẩn hóa Unicode NFC, xử lý khoảng trắng, giữ nguyên phân dòng
    - Ghép nối với thông tin Metadata tương ứng theo ID
    - Xuất dữ liệu theo cơ chế streaming với cơ chế ghi nguyên tử (atomic write)

Input:  data/raw/legal_documents/metadata/*.parquet
        data/raw/legal_documents/content/*.parquet
Output: data/processed/legal_documents/legal_documents_preprocessed.parquet
"""

import argparse
from pathlib import Path
import re
import sys
import unicodedata
from bs4 import BeautifulSoup
import pandas as pd
import pyarrow as pa
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
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "legal_documents"
META_DIR = RAW_DIR / "metadata"
CONTENT_DIR = RAW_DIR / "content"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "legal_documents"
OUTPUT_FILE = OUTPUT_DIR / "legal_documents_preprocessed.parquet"
TEMP_FILE = OUTPUT_DIR / "legal_documents_preprocessed.parquet.tmp"

SELECTED_META_COLS = [
    "id",
    "title",
    "so_ky_hieu",
    "loai_van_ban",
    "co_quan_ban_hanh",
    "ngay_ban_hanh",
    "ngay_co_hieu_luc",
    "tinh_trang_hieu_luc",
    "linh_vuc",
]

FINAL_COLS = [
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


def extract_clean_text_from_html(html_content: str) -> str:
    """Trích xuất và chuẩn hóa văn bản thuần từ chuỗi HTML."""
    if not html_content or not isinstance(html_content, str):
        return ""
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Triệt tiêu thẻ rác
    for tag in soup(["script", "style", "head", "meta", "title"]):
        tag.decompose()

    # 2. Bóc text ngăn cách các block bằng newline
    text = soup.get_text(separator="\n")

    # 3. Chuẩn hóa tiếng Việt và dấu ngắt dòng
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 4. Khử tàn dư thẻ vỡ do ký tự so sánh toán học
    text = re.sub(r"<\s*/?[a-zA-Z0-9_≤≥%]+(?:\s+[^>]*)?>", " ", text)

    # 5. Thu gọn space thừa, bảo toàn dòng ngắt cấu trúc
    lines = [
        re.sub(r"[^\S\r\n\t]+", " ", line).strip(" ") for line in text.split("\n")
    ]
    return "\n".join(line for line in lines if line)


def preprocess_legal_documents(force: bool = False) -> None:
    """Thực thi pipeline tiền xử lý cho Vietnamese Legal Documents."""
    if not META_DIR.exists() or not CONTENT_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu nguồn tại: {RAW_DIR}. "
            "Hãy đảm bảo script download_legal_documents.py đã được thực thi thành công."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not force and OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:
        file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
        print(
            f"Artifact đã tồn tại: {OUTPUT_FILE} ({file_size_mb:.2f} MB). "
            "Bỏ qua xử lý (Dùng --force để chạy lại)."
        )
        return

    print("=" * 80)
    print("PIPELINE: PREPROCESSING VIETNAMESE LEGAL DOCUMENTS")
    print("=" * 80)

    # 1. Nạp và kiểm tra nghiêm ngặt Metadata Shard
    meta_files = sorted(META_DIR.glob("*.parquet"))
    if len(meta_files) != 1:
        raise ValueError(
            f"Kỳ vọng chính xác 1 file metadata parquet, phát hiện {len(meta_files)} file."
        )

    meta_file = meta_files[0]
    print(f"Đang nạp Metadata từ: {meta_file.name}")
    meta_df = pd.read_parquet(meta_file)
    meta_df["id"] = meta_df["id"].astype(str).str.strip()

    available_meta = [c for c in SELECTED_META_COLS if c in meta_df.columns]
    meta_df = meta_df[available_meta].drop_duplicates(subset=["id"])
    for col in available_meta:
        meta_df[col] = meta_df[col].fillna("").astype(str)

    print(f"• Số lượng metadata documents hợp lệ: {len(meta_df):,}")

    # 2. Kiểm tra Content Partitions
    content_files = sorted(CONTENT_DIR.glob("*.parquet"))
    if not content_files:
        raise FileNotFoundError(f"Không tìm thấy partition content nào trong {CONTENT_DIR}")

    print(f"• Phát hiện {len(content_files)} file partition content.")

    # Dọn dẹp temp file cũ nếu có sót lại từ lần chạy lỗi trước
    if TEMP_FILE.exists():
        TEMP_FILE.unlink()

    parquet_writer = None
    total_processed = 0
    pipeline_success = False

    try:
        for i, file_path in enumerate(content_files, 1):
            print(f"[{i:02d}/{len(content_files):02d}] Đang xử lý: {file_path.name}...")
            part_df = pd.read_parquet(file_path)
            part_df["id"] = part_df["id"].astype(str).str.strip()

            # Trích xuất text và metadata phái sinh
            part_df["text"] = part_df["content_html"].apply(extract_clean_text_from_html)
            part_df["char_count"] = part_df["text"].str.len().fillna(0).astype(int)
            part_df["word_count"] = part_df["text"].str.split().str.len().fillna(0).astype(int)
            part_df["is_empty"] = part_df["char_count"] == 0

            # Merge với metadata
            merged_df = pd.merge(
                part_df[["id", "text", "char_count", "word_count", "is_empty"]],
                meta_df,
                on="id",
                how="left",
            )

            # Đồng bộ schema trước khi build table
            for col in FINAL_COLS:
                if col not in merged_df.columns:
                    merged_df[col] = "" if col not in ["char_count", "word_count", "is_empty"] else 0

            merged_df = merged_df[FINAL_COLS]

            for col in FINAL_COLS:
                if col in ["char_count", "word_count"]:
                    merged_df[col] = merged_df[col].fillna(0).astype("int64")
                elif col == "is_empty":
                    merged_df[col] = merged_df[col].astype("bool")
                else:
                    merged_df[col] = merged_df[col].fillna("").astype("string")

            table = pa.Table.from_pandas(merged_df, preserve_index=False)

            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(
                    str(TEMP_FILE), table.schema, compression="snappy"
                )
            parquet_writer.write_table(table)
            total_processed += len(merged_df)

        pipeline_success = True

    finally:
        # Đảm bảo đóng writer để flush buffer và nhả file handle trên Windows
        if parquet_writer is not None:
            parquet_writer.close()

        if pipeline_success:
            # Atomic replace file tạm sang output chính thức
            TEMP_FILE.replace(OUTPUT_FILE)
        else:
            # Xóa sạch file tạm nếu có exception xảy ra
            if TEMP_FILE.exists():
                TEMP_FILE.unlink(missing_ok=True)

    print("=" * 80)
    print("HOÀN TẤT TIỀN XỬ LÝ LEGAL DOCUMENTS!")
    print(f"• Tổng số văn bản đã trích xuất & lưu: {total_processed:,}")
    print(f"• Artifact output: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tiền xử lý và kết hợp Vietnamese Legal Documents thành file Parquet duy nhất."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Buộc xử lý lại toàn bộ ngay cả khi artifact đã tồn tại.",
    )
    args = parser.parse_args()
    preprocess_legal_documents(force=args.force)