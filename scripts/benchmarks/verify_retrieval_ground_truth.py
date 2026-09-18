# scripts/benchmarks/verify_retrieval_ground_truth.py
"""Script kiểm chứng tính toàn vẹn của Retrieval Ground Truth và ID Alignment.

Kiểm tra:
    1. Kiểm tra tồn tại và tính liên tục tuyệt đối của global_article_id trong corpus [0, 59635]
    2. Kiểm tra tính hợp lệ của query ID (qid not None, unique)
    3. Xác thực schema của relevant_laws (phải là list các số nguyên)
    4. Xác minh nhãn liên kết: không rỗng, không trùng nội tại, nằm trong dải ID corpus
    5. Thống kê phân phối ground truth (single-label vs multi-label, corpus coverage)
    6. Trả về mã thoát chuẩn (exit code 0 nếu hợp lệ, 1 nếu vi phạm)
"""

import argparse
import json
from pathlib import Path
import sys
from typing import List, Tuple
import numpy as np
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
DEFAULT_CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_retrieval"
    / "corpus_articles_preprocessed.parquet"
)
DEFAULT_TRAIN_PATH = (
    PROJECT_ROOT / "data" / "raw" / "legal_retrieval" / "train.json"
)

EXPECTED_CORPUS_DOCS = 59636


def verify_retrieval_ground_truth(
    corpus_path: Path,
    train_path: Path,
    strict: bool = True,
) -> int:
    """Thực thi kiểm chứng ground truth và ID alignment."""
    if not corpus_path.exists():
        print(f"Lỗi: Không tìm thấy file corpus: {corpus_path}")
        return 1
    if not train_path.exists():
        print(f"Lỗi: Không tìm thấy file train queries: {train_path}")
        return 1

    print("=" * 60)
    print("KIỂM CHỨNG RETRIEVAL GROUND TRUTH & ID ALIGNMENT")
    print("=" * 60)

    # 1. Đọc và kiểm chứng tính liên tục tuyệt đối của Corpus Article IDs
    print(f"Đang đọc Corpus: {corpus_path.name}...")
    corpus_table = pq.read_table(corpus_path, columns=["global_article_id"])
    corpus_ids = set(corpus_table["global_article_id"].to_pylist())
    total_corpus_docs = len(corpus_ids)

    if total_corpus_docs == 0:
        print("Lỗi: Corpus không có bản ghi nào!")
        return 1

    expected_corpus_ids = set(range(EXPECTED_CORPUS_DOCS))
    if corpus_ids != expected_corpus_ids:
        missing_ids = expected_corpus_ids - corpus_ids
        extra_ids = corpus_ids - expected_corpus_ids
        print(
            f"[FAIL] Corpus global_article_id không liên tục hoặc sai lệch dải ID. "
            f"Missing: {len(missing_ids)}, Extra: {len(extra_ids)}"
        )
        return 1

    min_corpus_id = min(corpus_ids)
    max_corpus_id = max(corpus_ids)
    print(f"  - Tổng số articles trong Corpus: {total_corpus_docs:,}")
    print(f"  - Dải ID Corpus: [{min_corpus_id}, {max_corpus_id}]")
    print("  - [PASS] 100% ID từ 0 đến 59,635 tồn tại liên tục.")

    # 2. Đọc Queries train.json
    print(f"\nĐang đọc queries: {train_path.name}...")
    with open(train_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    total_queries = len(queries)
    print(f"  - Tổng số queries: {total_queries:,}")

    if total_queries == 0:
        print("Lỗi: Tập query rỗng!")
        return 1

    # 3. Kiểm tra tính hợp lệ của query_id (qid)
    query_ids = [item.get("qid") for item in queries]
    if any(qid is None for qid in query_ids):
        print("[FAIL] Phát hiện query không có trường 'qid'!")
        return 1
    if len(query_ids) != len(set(query_ids)):
        print("[FAIL] Phát hiện 'qid' bị trùng lặp trong tập queries!")
        return 1

    # 4. Phân tích Ground Truth & ID Alignment
    has_error = False
    all_relevant_ids: List[int] = []
    labels_per_query: List[int] = []
    out_of_bounds_queries: List[Tuple[int, int]] = []
    empty_label_queries: List[int] = []
    duplicate_label_queries: List[int] = []

    for item in queries:
        qid = item.get("qid")
        labels = item.get("relevant_laws", [])

        # Kiểm tra kiểu dữ liệu của relevant_laws
        if not isinstance(labels, list):
            print(f"[FAIL] qid={qid}: 'relevant_laws' phải là list.")
            has_error = True
            continue

        if any(not isinstance(doc_id, int) for doc_id in labels):
            print(f"[FAIL] qid={qid}: relevant_laws chứa ID không phải integer.")
            has_error = True
            continue

        if not labels:
            empty_label_queries.append(qid)
            labels_per_query.append(0)
            continue

        labels_per_query.append(len(labels))

        if len(labels) != len(set(labels)):
            duplicate_label_queries.append(qid)

        for doc_id in labels:
            all_relevant_ids.append(doc_id)
            if doc_id not in corpus_ids:
                out_of_bounds_queries.append((qid, doc_id))

    # 5. Báo cáo thống kê và tổng hợp kết quả
    labels_arr = np.array(labels_per_query)
    unique_relevant_ids = set(all_relevant_ids)

    print("\n" + "-" * 25 + " KẾT QUẢ KIỂM CHỨNG " + "-" * 25)
    print(f"1. Tổng số query: {total_queries:,}")
    print(f"2. Query không có nhãn (empty labels): {len(empty_label_queries)}")
    print(f"3. Query có nhãn trùng lặp nội tại: {len(duplicate_label_queries)}")
    print(f"4. Nhãn nằm ngoài biên Corpus (Out of bounds): {len(out_of_bounds_queries)}")

    if out_of_bounds_queries:
        has_error = True
        print("   [FAIL] Danh sách (qid, invalid_doc_id) vi phạm:")
        for q, d in out_of_bounds_queries[:5]:
            print(f"     - qid={q} -> doc_id={d} không tồn tại trong Corpus!")
    else:
        print("   [PASS] 100% ID nhãn khớp hoàn toàn với Corpus [0, 59635].")

    if empty_label_queries:
        if strict:
            has_error = True
            print(f"   [FAIL] Có {len(empty_label_queries)} query rỗng nhãn (strict mode).")
        else:
            print(f"   [WARN] Có {len(empty_label_queries)} query rỗng nhãn.")

    if len(labels_arr) > 0:
        print("\n5. Thống kê phân phối số nhãn / query:")
        print(f"  - Min số nhãn / query: {np.min(labels_arr)}")
        print(f"  - 25th percentile: {np.percentile(labels_arr, 25)}")
        print(f"  - Median: {np.median(labels_arr)}")
        print(f"  - Mean: {np.mean(labels_arr):.2f}")
        print(f"  - 75th percentile: {np.percentile(labels_arr, 75)}")
        print(f"  - Max số nhãn / query: {np.max(labels_arr)}")

        single_label = int(np.sum(labels_arr == 1))
        multi_label = int(np.sum(labels_arr > 1))
        print("\n6. Phân loại bài toán Retrieval:")
        print(
            f"  - Single-label queries: {single_label:,} ({single_label / total_queries * 100:.1f}%)"
        )
        print(
            f"  - Multi-label queries: {multi_label:,} ({multi_label / total_queries * 100:.1f}%)"
        )
        print(f"  - Tổng số nhãn liên kết: {len(all_relevant_ids):,}")
        print(
            f"  - Độ phủ Corpus (Coverage): {len(unique_relevant_ids):,} / {total_corpus_docs:,} "
            f"({len(unique_relevant_ids) / total_corpus_docs * 100:.2f}%)"
        )

    print("=" * 60)
    if has_error:
        print("KẾT QUẢ: VERIFICATION THẤT BẠI.")
        return 1

    print("KẾT QUẢ: VERIFICATION THÀNH CÔNG.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kiểm chứng Retrieval Ground Truth và ID Alignment với Corpus."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Đường dẫn file corpus_articles_preprocessed.parquet.",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="Đường dẫn file train.json queries.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Không coi empty label queries là lỗi nghiêm trọng.",
    )
    args = parser.parse_args()

    exit_code = verify_retrieval_ground_truth(
        corpus_path=args.corpus,
        train_path=args.train,
        strict=not args.no_strict,
    )
    sys.exit(exit_code)