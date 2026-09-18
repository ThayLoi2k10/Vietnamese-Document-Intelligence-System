from pathlib import Path
from datasets import Dataset


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data")


# ============================================================
# FIND ALL .ARROW FILES INSIDE data/
# ============================================================

arrow_files = list(DATA_DIR.rglob("*.arrow"))

if not arrow_files:
    print("Không tìm thấy file .arrow nào trong thư mục data/")
    exit()


print("=" * 60)
print(f"Tìm thấy {len(arrow_files)} file .arrow")
print("=" * 60)


# ============================================================
# CONVERT EACH ARROW FILE TO PARQUET
# ============================================================

for arrow_file in arrow_files:

    print("\n" + "-" * 60)
    print(f"Đang xử lý: {arrow_file}")

    try:
        # ----------------------------------------------------
        # Read Arrow file using Hugging Face Dataset
        # ----------------------------------------------------

        dataset = Dataset.from_file(str(arrow_file))

        print(f"Rows: {len(dataset)}")
        print(f"Columns: {dataset.column_names}")

        # ----------------------------------------------------
        # Create output path
        # Example:
        #
        # data/raw/legal_documents/metadata/data.arrow
        #
        # becomes:
        #
        # data/raw/legal_documents/metadata/data.parquet
        # ----------------------------------------------------

        parquet_file = arrow_file.with_suffix(".parquet")

        # ----------------------------------------------------
        # Convert to Parquet
        # ----------------------------------------------------

        dataset.to_parquet(str(parquet_file))

        print(f"Đã tạo: {parquet_file}")

        # ----------------------------------------------------
        # Verify Parquet file
        # ----------------------------------------------------

        check_dataset = Dataset.from_parquet(str(parquet_file))

        if len(check_dataset) != len(dataset):
            print("LỖI: số rows không khớp!")
            continue

        if check_dataset.column_names != dataset.column_names:
            print("LỖI: columns không khớp!")
            continue

        print("VERIFY: PASS")

        # ----------------------------------------------------
        # Delete original Arrow file
        # ----------------------------------------------------

        arrow_file.unlink()

        print(f"Đã xóa: {arrow_file}")

    except Exception as e:

        print(f"LỖI khi xử lý {arrow_file}")
        print(e)


# ============================================================
# FINAL CHECK
# ============================================================

remaining_arrow = list(DATA_DIR.rglob("*.arrow"))
parquet_files = list(DATA_DIR.rglob("*.parquet"))

print("\n" + "=" * 60)
print("CONVERSION COMPLETE")
print("=" * 60)

print(f"Remaining .arrow files: {len(remaining_arrow)}")
print(f"Total .parquet files: {len(parquet_files)}")

if remaining_arrow:
    print("\nCác file .arrow còn lại:")

    for file in remaining_arrow:
        print(f"  - {file}")

else:
    print("\nPASS: Không còn file .arrow trong data/")