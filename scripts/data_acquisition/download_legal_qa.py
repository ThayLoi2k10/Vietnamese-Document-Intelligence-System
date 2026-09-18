# scripts/data_acquisition/download_legal_qa.py
"""Script tải dữ liệu Vietnamese Legal QA từ Hugging Face Hub và xuất sang Parquet.

Dataset: thangvip/vietnamese-legal-qa
Split: train
Output: data/raw/legal_qa/vietnamese_legal_qa.parquet
"""

import argparse
from pathlib import Path
import sys
from datasets import load_dataset

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
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "legal_qa"
OUTPUT_FILE = OUTPUT_DIR / "vietnamese_legal_qa.parquet"

REPO_ID = "thangvip/vietnamese-legal-qa"


def download_legal_qa(force: bool = False) -> None:
    """Tải và lưu dataset Vietnamese Legal QA dưới dạng Parquet."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Kiểm tra artifact đã tồn tại và hợp lệ
    if not force and OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:
        file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
        print(
            f"File artifact đã tồn tại: {OUTPUT_FILE} ({file_size_mb:.2f} MB). "
            "Bỏ qua tải (Dùng --force để tải lại)."
        )
        return

    print("=" * 60)
    print("Loading Vietnamese Legal QA from Hugging Face")
    print("=" * 60)

    try:
        dataset = load_dataset(REPO_ID, split="train")
        if len(dataset) == 0:
            raise ValueError("Dataset rỗng, không thể lưu artifact.")

        print("\nDataset loaded successfully!")
        print(f"Tổng số bản ghi: {len(dataset):,}")
        print(f"Các cột dữ liệu: {dataset.column_names}")

        print("\n" + "=" * 60)
        print("Saving dataset as Parquet")
        print("=" * 60)

        dataset.to_parquet(str(OUTPUT_FILE))

        file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
        print("\nParquet file created successfully!")
        print(f"Đường dẫn: {OUTPUT_FILE}")
        print(f"Kích thước: {file_size_mb:.2f} MB")
        print("\n" + "=" * 60)
        print("DONE")
        print("=" * 60)

    except Exception as e:
        print(f"Lỗi khi tải hoặc lưu dataset '{REPO_ID}': {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Thu thập dữ liệu Vietnamese Legal QA từ Hugging Face và lưu dạng Parquet."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Buộc tải lại dữ liệu ngay cả khi file Parquet đã tồn tại.",
    )
    args = parser.parse_args()
    download_legal_qa(force=args.force)