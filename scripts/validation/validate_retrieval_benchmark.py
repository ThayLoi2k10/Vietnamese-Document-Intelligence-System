# scripts/validation/validate_retrieval_benchmark.py
"""Script kiểm định tính toàn vẹn của Retrieval Benchmark Dataset.

Kiểm tra:
    1. Kiểm tra tồn tại dữ liệu nguồn và schema của từng split (Train / Test)
    2. Kiểm định từng query: qid hợp lệ, text không rỗng, nhãn integer, sorted & không lặp
    3. Xác minh 100% article IDs nằm trong tập ID của Corpus
    4. Kiểm tra tính phân hoạch toàn vẹn:
       - Rời rạc tuyệt đối: Train ∩ Test = ∅ (Zero leakage)
       - Bảo toàn dữ liệu: Train ∪ Test = Raw All Queries (xác thực defensive trên raw JSON)
    5. Xuất báo cáo kiểm định và trả về exit code cho CI/CD
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple
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
DEFAULT_BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmarks"
DEFAULT_TRAIN_PATH = DEFAULT_BENCHMARK_DIR / "retrieval_train.json"
DEFAULT_TEST_PATH = DEFAULT_BENCHMARK_DIR / "retrieval_test.json"
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "legal_retrieval" / "train.json"
DEFAULT_CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_retrieval"
    / "corpus_articles_preprocessed.parquet"
)
DEFAULT_REPORT_PATH = DEFAULT_BENCHMARK_DIR / "retrieval_validation_report.txt"

REQUIRED_FIELDS = {"query_id", "query_text", "relevant_article_ids"}


def validate_split(
    data: List[Dict[str, Any]],
    split_name: str,
    corpus_ids: Set[int],
    logs: List[str],
) -> Tuple[bool, Set[int]]:
    """Kiểm tra tính hợp lệ nội bộ của một tập split."""
    logs.append(f"\n--- KIỂM ĐỊNH TẬP: {split_name} ---")
    if not isinstance(data, list) or len(data) == 0:
        logs.append(f"[FAIL] {split_name} rỗng hoặc không phải định dạng list!")
        return False, set()

    is_valid = True
    qids: List[int] = []
    total_labels = 0

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            logs.append(f"[FAIL] Bản ghi index {idx} không phải là dict!")
            is_valid = False
            continue

        # 1. Kiểm tra Schema
        if set(item.keys()) != REQUIRED_FIELDS:
            logs.append(
                f"[FAIL] Bản ghi index {idx} sai schema: {list(item.keys())} "
                f"(kỳ vọng: {REQUIRED_FIELDS})"
            )
            is_valid = False
            continue

        qid = item["query_id"]
        qtext = item["query_text"]
        labels = item["relevant_article_ids"]

        # 2. Kiểm tra query_id
        if not isinstance(qid, int):
            logs.append(f"[FAIL] query_id tại index {idx} không phải số nguyên: {qid}")
            is_valid = False
        else:
            qids.append(qid)

        # 3. Kiểm tra query_text
        if not isinstance(qtext, str) or not qtext.strip():
            logs.append(f"[FAIL] query_id={qid} có query_text rỗng hoặc không phải chuỗi!")
            is_valid = False

        # 4. Kiểm tra nhãn liên kết
        if not isinstance(labels, list) or len(labels) == 0:
            logs.append(f"[FAIL] query_id={qid} có relevant_article_ids rỗng!")
            is_valid = False
            continue

        # Chặn sớm nếu phần tử không phải int để bảo vệ set() và sorted() phía sau
        if any(not isinstance(doc_id, int) for doc_id in labels):
            logs.append(f"[FAIL] query_id={qid} chứa nhãn không phải kiểu integer!")
            is_valid = False
            continue

        if len(labels) != len(set(labels)):
            logs.append(f"[FAIL] query_id={qid} bị trùng lặp nhãn nội tại: {labels}")
            is_valid = False

        if labels != sorted(labels):
            logs.append(f"[FAIL] query_id={qid} nhãn chưa được sắp xếp tăng dần: {labels}")
            is_valid = False

        # 5. Kiểm tra ID có thuộc Corpus không
        invalid_ids = [doc_id for doc_id in labels if doc_id not in corpus_ids]
        if invalid_ids:
            logs.append(f"[FAIL] query_id={qid} chứa ID nằm ngoài Corpus: {invalid_ids}")
            is_valid = False

        total_labels += len(labels)

    # 6. Kiểm tra tính duy nhất của query_id trong tập
    duplicate_qids = len(qids) - len(set(qids))
    if duplicate_qids > 0:
        logs.append(f"[FAIL] {split_name} có {duplicate_qids} query_id bị trùng lặp!")
        is_valid = False

    avg_labels = total_labels / len(data) if len(data) > 0 else 0.0
    logs.append(f"  * Tổng số queries          : {len(data):,}")
    logs.append(f"  * Tổng số cặp nhãn (pairs) : {total_labels:,}")
    logs.append(f"  * Trung bình nhãn / query  : {avg_labels:.2f}")
    logs.append(f"  * Trạng thái kiểm tra nội bộ: {'PASS' if is_valid else 'FAIL'}")

    return is_valid, set(qids)


def validate_retrieval_benchmark(
    train_path: Path,
    test_path: Path,
    raw_path: Path,
    corpus_path: Path,
    report_path: Path,
) -> int:
    """Thực thi kiểm định toàn diện tập Retrieval Benchmark."""
    logs: List[str] = []
    logs.append("=" * 60)
    logs.append("KIỂM ĐỊNH TOÀN VẸN: RETRIEVAL BENCHMARK DATASET")
    logs.append("=" * 60)

    # 1. Đọc Corpus ID Set và guard rỗng trước khi truy cập min/max
    if not corpus_path.exists():
        logs.append(f"[CRITICAL] Không tìm thấy corpus tại: {corpus_path}")
        print("\n".join(logs))
        return 1

    corpus_table = pq.read_table(corpus_path, columns=["global_article_id"])
    corpus_ids = set(corpus_table["global_article_id"].to_pylist())

    if not corpus_ids:
        logs.append("[CRITICAL] Corpus không chứa global_article_id nào.")
        print("\n".join(logs))
        return 1

    logs.append(
        f"Corpus size tham chiếu: {len(corpus_ids):,} articles "
        f"[{min(corpus_ids)}, {max(corpus_ids)}]"
    )

    # 2. Đọc Train & Test JSON
    if not train_path.exists() or not test_path.exists():
        logs.append(
            f"[CRITICAL] Thiếu file benchmark: {train_path.name} hoặc {test_path.name} không tồn tại!"
        )
        print("\n".join(logs))
        return 1

    try:
        with open(train_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
    except Exception as e:
        logs.append(f"[CRITICAL] Không thể parse file JSON benchmark: {e}")
        print("\n".join(logs))
        return 1

    # 3. Validate từng tập độc lập
    train_valid, train_qids = validate_split(train_data, "TRAIN", corpus_ids, logs)
    test_valid, test_qids = validate_split(test_data, "TEST", corpus_ids, logs)

    # 4. Kiểm tra rò rỉ ranh giới (Disjoint check: Train ∩ Test = ∅)
    logs.append("\n--- KIỂM TRA PHÂN HOẠCH TẬP HỢP (PARTITION INTEGRITY) ---")
    overlap = train_qids.intersection(test_qids)
    leakage_valid = len(overlap) == 0

    if not leakage_valid:
        logs.append(f"[FAIL] Rò rỉ query_id giữa Train và Test ({len(overlap)} IDs vi phạm)!")
    else:
        logs.append("[PASS] Train ∩ Test = ∅ (0% data leakage về query_id).")

    # 5. Kiểm tra tính bao phủ toàn vẹn có bảo vệ cấu trúc (Train ∪ Test = Raw All Queries)
    partition_complete = True
    if raw_path.exists():
        try:
            with open(raw_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except Exception as e:
            logs.append(f"[FAIL] Không thể đọc raw benchmark: {e}")
            partition_complete = False
        else:
            if not isinstance(raw_data, list):
                logs.append("[FAIL] Raw benchmark không có top-level dạng list.")
                partition_complete = False
            elif any(not isinstance(item, dict) for item in raw_data):
                logs.append("[FAIL] Raw benchmark chứa bản ghi không phải dict.")
                partition_complete = False
            else:
                raw_qids = {
                    item["qid"]
                    for item in raw_data
                    if item.get("qid") is not None
                }
                combined_qids = train_qids | test_qids
                if combined_qids != raw_qids:
                    missing_from_splits = raw_qids - combined_qids
                    extra_in_splits = combined_qids - raw_qids
                    logs.append(
                        f"[FAIL] Train ∪ Test không khớp 100% với file nguồn raw train.json!\n"
                        f"       Thiếu trong splits: {len(missing_from_splits)}, Dư thừa: {len(extra_in_splits)}"
                    )
                    partition_complete = False
                else:
                    logs.append(
                        f"[PASS] Train ∪ Test = Raw All ({len(raw_qids):,} queries được bảo toàn 100%)."
                    )
    else:
        logs.append(f"[WARN] Không tìm thấy raw file {raw_path.name} để kiểm tra tính toàn vẹn hợp nhất.")

    # 6. Kiểm tra tỷ lệ kích thước (Kỳ vọng ~80% Train, ~20% Test)
    total_split_queries = len(train_data) + len(test_data)
    test_ratio = len(test_data) / total_split_queries if total_split_queries > 0 else 0
    ratio_valid = 0.18 <= test_ratio <= 0.22

    if not ratio_valid:
        logs.append(
            f"[FAIL] Tỷ lệ phân tập lệch chuẩn 80/20 (Test ratio: {test_ratio * 100:.2f}%)"
        )
    else:
        logs.append(
            f"[PASS] Phân bổ tỷ lệ kích thước hợp lệ: Train={len(train_data):,} "
            f"({(1 - test_ratio) * 100:.1f}%), Test={len(test_data):,} ({test_ratio * 100:.1f}%)."
        )

    # 7. Tổng kết & Ghi báo cáo
    all_passed = (
        train_valid
        and test_valid
        and leakage_valid
        and partition_complete
        and ratio_valid
    )

    logs.append("\n" + "=" * 60)
    logs.append(
        f"KẾT QUẢ CUỐI CÙNG: {'ALL ASSERTIONS PASSED (100%)' if all_passed else 'FAILED'}"
    )
    logs.append("=" * 60)

    report_content = "\n".join(logs)
    print(report_content)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content + "\n")
    print(f"\nĐã ghi báo cáo: {report_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kiểm định tính toàn vẹn và không rò rỉ của Retrieval Benchmark."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="Đường dẫn file retrieval_train.json.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=DEFAULT_TEST_PATH,
        help="Đường dẫn file retrieval_test.json.",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help="Đường dẫn file raw train.json.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Đường dẫn file corpus_articles_preprocessed.parquet.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Đường dẫn lưu file báo cáo kiểm định.",
    )
    args = parser.parse_args()

    exit_code = validate_retrieval_benchmark(
        train_path=args.train,
        test_path=args.test,
        raw_path=args.raw,
        corpus_path=args.corpus,
        report_path=args.report,
    )
    sys.exit(exit_code)