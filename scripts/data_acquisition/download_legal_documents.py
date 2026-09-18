# scripts/data_acquisition/download_legal_documents.py
"""Script tải dữ liệu Vietnamese Legal Documents từ Hugging Face Hub và xuất sang Parquet.

Dataset: th1nhng0/vietnamese-legal-documents
Configs:
    - metadata: 1 shard Parquet
    - content: 12 shards Parquet
    - relationships: 1 shard Parquet
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
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "legal_documents"

# Cấu hình tải và số lượng shard tương ứng với raw artifact chuẩn
CONFIG_SHARDS = {
    "metadata": 1,
    "content": 12,
    "relationships": 1,
}


def download_legal_documents(force: bool = False) -> None:
    """Tải và lưu trực tiếp các configs của Vietnamese Legal Documents dưới dạng Parquet."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for config, expected_shards in CONFIG_SHARDS.items():
        print("\n" + "=" * 60)
        print(f"Downloading config: {config}")
        print("=" * 60)

        config_dir = OUTPUT_DIR / config
        config_dir.mkdir(parents=True, exist_ok=True)

        expected_files = [
            config_dir / f"data-{idx:05d}-of-{expected_shards:05d}.parquet"
            for idx in range(expected_shards)
        ]
        all_expected_exist = all(path.exists() for path in expected_files)

        if not force and all_expected_exist:
            print(
                f"Thư mục {config_dir} đã có đủ {expected_shards} file Parquet "
                "đúng cấu trúc. Bỏ qua tải (Dùng --force để tải lại)."
            )
            continue

        try:
            dataset = load_dataset(
                "th1nhng0/vietnamese-legal-documents",
                config,
                split="data",
            )

            if expected_shards == 1:
                out_file = expected_files[0]
                dataset.to_parquet(str(out_file))
                print(f"Đã lưu: {out_file.name}")
            else:
                for idx in range(expected_shards):
                    out_file = expected_files[idx]
                    shard = dataset.shard(
                        num_shards=expected_shards, index=idx, contiguous=True
                    )
                    shard.to_parquet(str(out_file))
                    print(f"Đã lưu shard [{idx + 1}/{expected_shards}]: {out_file.name}")

            print(f"Hoàn thành config '{config}' với tổng số {len(dataset):,} bản ghi.")
        except Exception as e:
            print(f"Lỗi khi tải hoặc lưu config '{config}': {e}")
            raise

    print("\nQuá trình thu thập Vietnamese Legal Documents hoàn tất.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Thu thập dữ liệu pháp luật tiếng Việt từ Hugging Face và xuất dạng Parquet."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Buộc tải lại toàn bộ ngay cả khi các file Parquet đã tồn tại.",
    )
    args = parser.parse_args()
    download_legal_documents(force=args.force)