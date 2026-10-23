"""Build and freeze unified benchmark dataset for Phase 10 system comparison."""


import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List

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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def atomic_write_json(file_path: Path, data: Any) -> None:
    """Ghi file JSON an toàn thông qua file tạm và atomic replace."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=file_path.parent, delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp_name = tmp.name
    os.replace(tmp_name, file_path)


def build_phase10_comparison_dataset(
    retrieval_train_path: Path = PROJECT_ROOT / "data" / "benchmarks" / "retrieval_train.json",
    retrieval_test_path: Path = PROJECT_ROOT / "data" / "benchmarks" / "retrieval_test.json",
    phase9_preds_path: Path = (
        PROJECT_ROOT
        / "artifacts"
        / "phase9"
        / "PROJECT2_PHASE9"
        / "benchmarks"
        / "exp_phase9_baseline_predictions.json"
    ),
    phase9_fails_path: Path = (
        PROJECT_ROOT
        / "artifacts"
        / "phase9"
        / "PROJECT2_PHASE9"
        / "benchmarks"
        / "exp_phase9_baseline_failures.json"
    ),
    output_path: Path = PROJECT_ROOT / "data" / "benchmarks" / "phase10_comparison_dataset.json",
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Tạo và xác minh tập dữ liệu đối chuẩn Phase 10."""
    print("=" * 80)
    print("XÂY DỰNG TẬP DỮ LIỆU ĐỐI CHUẨN PHASE 10 (COMPARISON DATASET)")
    print("=" * 80)

    if output_path.exists() and not force:
        print(f"[THÔNG BÁO] Tệp dữ liệu đã tồn tại: {output_path}")
        print("Sử dụng --force để xây dựng lại.")
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 1. Nạp tập retrieval_train gốc
    print(f"[1/5] Nạp 1,401 câu hỏi DEVELOPMENT split từ: {retrieval_train_path}...")
    with open(retrieval_train_path, "r", encoding="utf-8") as f:
        all_train_queries = json.load(f)

    dev_queries = all_train_queries[:1401]
    if len(dev_queries) != 1401:
        raise ValueError(
            f"Kỳ vọng đúng 1,401 câu hỏi DEVELOPMENT, nhận được {len(dev_queries)}"
        )

    # 2. Nạp dữ liệu đối soát Phase 9
    print("[2/5] Nạp và đối soát với Phase 9 baseline predictions & failures...")
    with open(phase9_preds_path, "r", encoding="utf-8") as f:
        p9_preds = json.load(f)
    with open(phase9_fails_path, "r", encoding="utf-8") as f:
        p9_fails = json.load(f)

    print(f"      - Phase 9 predictions: {len(p9_preds)} mẫu")
    print(f"      - Phase 9 failures:    {len(p9_fails)} mẫu")
    total_p9 = len(p9_preds) + len(p9_fails)
    if total_p9 != 1401:
        raise ValueError(f"Tổng số mẫu Phase 9 không khớp 1,401 (nhận được {total_p9})")

    p9_lookup: Dict[int, str] = {}
    for p in p9_preds:
        p9_lookup[p["query_id"]] = p["query"]
    for f in p9_fails:
        p9_lookup[f["query_id"]] = f["query"]

    # 3. Kiểm tra tính toàn vẹn 1-1
    print("[3/5] Xác minh khớp 100% ID và nội dung truy vấn với Phase 9...")
    mismatch_count = 0
    comparison_dataset = []

    for idx, q in enumerate(dev_queries):
        qid = q["query_id"]
        qtext = q["query_text"]
        gt_ids = q.get("relevant_article_ids", [])

        if qid not in p9_lookup:
            raise KeyError(f"Query ID {qid} không tìm thấy trong Phase 9 baseline!")

        if p9_lookup[qid] != qtext:
            mismatch_count += 1
            print(f"[CẢNH BÁO] Lệch văn bản tại Query ID {qid}")

        difficulty = "single_label" if len(gt_ids) == 1 else "multi_label"
        item = {
            "query_id": qid,
            "query_text": qtext,
            "relevant_article_ids": gt_ids,
            "difficulty": difficulty,
            "relevant_article_count": len(gt_ids),
            "question_type": "legal_retrieval_qa",
            "split_role": "DEVELOPMENT",
            "phase9_status": "COMPLETED" if qid in {p["query_id"] for p in p9_preds} else "FAILED",
        }
        comparison_dataset.append(item)

    if mismatch_count > 0:
        raise ValueError(f"Phát hiện {mismatch_count} mẫu bị sai lệch văn bản truy vấn!")
    print("      -> Toàn bộ 1,401 câu hỏi khớp chính xác 100% với Phase 9 baseline.")

    # 4. Kiểm tra rò rỉ dữ liệu (Zero Leakage Check)
    print(f"[4/5] Kiểm tra rò rỉ dữ liệu với FROZEN_TEST ({retrieval_test_path})...")
    with open(retrieval_test_path, "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    test_ids = {q["query_id"] for q in test_queries}
    test_texts = {q["query_text"].strip().lower() for q in test_queries}
    dev_ids = {q["query_id"] for q in dev_queries}
    dev_texts = {q["query_text"].strip().lower() for q in dev_queries}

    id_leak = dev_ids.intersection(test_ids)
    text_leak = dev_texts.intersection(test_texts)

    if id_leak:
        raise ValueError(f"RÒ RỈ DỮ LIỆU: Phát hiện {len(id_leak)} ID trùng với FROZEN_TEST: {id_leak}")
    if text_leak:
        raise ValueError(f"RÒ RỈ DỮ LIỆU: Phát hiện {len(text_leak)} câu hỏi trùng với FROZEN_TEST!")
    print("      -> Zero-leakage xác nhận: Không có trùng lặp ID hoặc nội dung với FROZEN_TEST.")

    # Thống kê phân bố độ khó
    single_count = sum(1 for q in comparison_dataset if q["difficulty"] == "single_label")
    multi_count = sum(1 for q in comparison_dataset if q["difficulty"] == "multi_label")
    print("\n      Thống kê tập dữ liệu đối chuẩn Phase 10:")
    print(f"      - Tổng số câu hỏi:        {len(comparison_dataset)}")
    print(f"      - Đơn nhãn (single-label): {single_count} ({single_count / len(comparison_dataset) * 100:.2f}%)")
    print(f"      - Đa nhãn (multi-label):   {multi_count} ({multi_count / len(comparison_dataset) * 100:.2f}%)")

    # 5. Ghi tệp đóng băng
    print(f"\n[5/5] Lưu tệp đóng băng: {output_path}...")
    atomic_write_json(output_path, comparison_dataset)
    print(f"      -> Thành công! Kích thước: {output_path.stat().st_size:,} bytes")
    print("=" * 80)

    return comparison_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Xây dựng tập dữ liệu đối chuẩn Phase 10 từ Phase 9 baseline."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ghi đè file nếu đã tồn tại",
    )
    args = parser.parse_args()

    build_phase10_comparison_dataset(force=args.force)


if __name__ == "__main__":
    main()
