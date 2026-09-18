import glob
import os
import sys
import pyarrow.dataset as ds
import pandas as pd
from bs4 import BeautifulSoup

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


def resolve_path(rel_path: str) -> str:
    candidates = [
        rel_path,
        os.path.join("..", rel_path),
        os.path.join("..", "..", rel_path),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError(f"Không tìm thấy: {rel_path}")


def main():
    print("=" * 80)
    print("BƯỚC 4.1: KHẢO SÁT CHI TIẾT RAW LEGAL DOCUMENTS")
    print("=" * 80)

    # 1. Quét danh sách file .parquet, loại bỏ file .json
    content_dir = resolve_path("data/raw/legal_documents/content")
    meta_dir = resolve_path("data/raw/legal_documents/metadata")
    rel_dir = resolve_path("data/raw/legal_documents/relationships")

    content_files = sorted(glob.glob(os.path.join(content_dir, "*.parquet")))
    meta_files = sorted(glob.glob(os.path.join(meta_dir, "*.parquet")))
    rel_files = sorted(glob.glob(os.path.join(rel_dir, "*.parquet")))

    print(f"• Số lượng file Parquet Content       : {len(content_files)}")
    print(f"• Số lượng file Parquet Metadata      : {len(meta_files)}")
    print(f"• Số lượng file Parquet Relationships : {len(rel_files)}")

    # 2. Kiểm tra Content Dataset
    content_ds = ds.dataset(content_files, format="parquet")
    total_content_rows = content_ds.count_rows()
    print(f"\n[1] CONTENT DATASET")
    print(f"• Tổng số bản ghi content : {total_content_rows:,}")
    print(f"• Schema columns          : {content_ds.schema.names}")

    # Lấy 1 bản ghi mẫu để giải phẫu HTML
    sample_batch = next(content_ds.to_batches(batch_size=1)).to_pandas().iloc[0]
    sample_id = sample_batch["id"]
    raw_html = str(sample_batch["content_html"])
    print(f"• Mẫu record ID đầu tiên  : {sample_id}")
    print(f"• Kích thước HTML thô     : {len(raw_html):,} ký tự")

    # Bóc tách thử cấu trúc thẻ HTML
    soup = BeautifulSoup(raw_html[:5000], "html.parser")
    tags = set(tag.name for tag in soup.find_all())
    print(f"• Các thẻ HTML chính xuất hiện: {tags}")
    print("• Đoạn HTML trích đoạn:")
    print(repr(raw_html[:300]))

    # 3. Kiểm tra Metadata
    print(f"\n[2] METADATA DATASET")
    meta_df = pd.read_parquet(meta_files[0])
    print(f"• Tổng số bản ghi metadata: {len(meta_df):,}")
    print(f"• Columns: {meta_df.columns.tolist()}")
    print("• Mẫu 1 record metadata:")
    print(meta_df.iloc[0].to_dict())

    # 4. Kiểm tra Relationships
    print(f"\n[3] RELATIONSHIPS DATASET")
    rel_df = pd.read_parquet(rel_files[0])
    print(f"• Tổng số bản ghi relationships: {len(rel_df):,}")
    print(f"• Columns: {rel_df.columns.tolist()}")
    print("• Mẫu 1 record relationship:")
    print(rel_df.iloc[0].to_dict())

    # 5. Kiểm tra tính tương thích của khóa (Key Alignment)
    print("\n[4] KIỂM TRA KHÓA LIÊN KẾT (JOIN KEYS)")
    content_sample_ids = next(content_ds.to_batches(batch_size=1000)).to_pandas()["id"]
    meta_ids = set(meta_df["id"].dropna().unique()) if "id" in meta_df.columns else set()
    
    match_count = content_sample_ids.isin(meta_ids).sum()
    print(f"• Tỷ lệ khớp ID mẫu (1,000 content IDs đầu tiên với metadata): {match_count}/1,000")
    print("=" * 80)


if __name__ == "__main__":
    main()