"""Validate integrity, boundaries, and zero-leakage constraints of the Phase 10 benchmark dataset."""

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
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=file_path.parent, delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp_name = tmp.name
    os.replace(tmp_name, file_path)


def validate_phase10_dataset(
    dataset_path: Path = PROJECT_ROOT / "data" / "benchmarks" / "phase10_comparison_dataset.json",
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
    report_output_path: Path = (
        PROJECT_ROOT / "data" / "benchmarks" / "phase10_dataset_validation_report.json"
    ),
) -> bool:
    print("=" * 80)
    print("KIỂM ĐỊNH TÍNH TOÀN VẸN TẬP DỮ LIỆU ĐỐI CHUẨN PHASE 10")
    print(f"Dataset path: {dataset_path}")
    print("=" * 80)

    report: Dict[str, Any] = {
        "status": "PASSED",
        "dataset_path": str(dataset_path),
        "checks": {},
        "summary": {},
        "errors": [],
    }

    if not dataset_path.exists():
        err = f"Không tìm thấy file tập dữ liệu: {dataset_path}"
        print(f"[THẤT BẠI] {err}")
        report["status"] = "FAILED"
        report["errors"].append(err)
        atomic_write_json(report_output_path, report)
        return False

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 1. Query count check
    query_count = len(dataset)
    check1_passed = (query_count == 1401)
    report["checks"]["query_count_check"] = {
        "passed": check1_passed,
        "observed_count": query_count,
        "expected_count": 1401,
    }
    if not check1_passed:
        report["errors"].append(f"Query count sai lệch: {query_count} != 1401")
    print(f"[{'PASS' if check1_passed else 'FAIL'}] 1. Query Count Check: {query_count} == 1,401")

    # 2. Check query IDs against Phase 9
    with open(phase9_preds_path, "r", encoding="utf-8") as f:
        p9_preds = json.load(f)
    with open(phase9_fails_path, "r", encoding="utf-8") as f:
        p9_fails = json.load(f)

    p9_pred_ids = {p["query_id"] for p in p9_preds}
    p9_fail_ids = {f["query_id"] for f in p9_fails}
    p9_all_ids = p9_pred_ids.union(p9_fail_ids)

    dataset_ids = [q["query_id"] for q in dataset]
    dataset_ids_set = set(dataset_ids)

    check2_no_dup = (len(dataset_ids) == len(dataset_ids_set))
    check2_ids_match = (dataset_ids_set == p9_all_ids)
    check2_passed = check2_no_dup and check2_ids_match

    report["checks"]["query_ids_check"] = {
        "passed": check2_passed,
        "unique_ids_count": len(dataset_ids_set),
        "phase9_ids_count": len(p9_all_ids),
        "missing_from_dataset": list(p9_all_ids - dataset_ids_set),
        "extra_in_dataset": list(dataset_ids_set - p9_all_ids),
    }
    if not check2_passed:
        report["errors"].append("Query IDs không khớp 100% với Phase 9 hoặc có ID trùng lặp")
    print(
        f"[{'PASS' if check2_passed else 'FAIL'}] 2. Query IDs Check: "
        f"{len(dataset_ids_set)} unique IDs khớp 100% Phase 9 (1,379 completed + 22 failed)"
    )

    # 3. Check question text fidelity
    p9_text_lookup = {}
    for p in p9_preds:
        p9_text_lookup[p["query_id"]] = p["query"]
    for f in p9_fails:
        p9_text_lookup[f["query_id"]] = f["query"]

    text_mismatches = []
    for q in dataset:
        qid = q["query_id"]
        qtext = q["query_text"]
        if qid in p9_text_lookup and p9_text_lookup[qid] != qtext:
            text_mismatches.append({"query_id": qid, "expected": p9_text_lookup[qid], "found": qtext})

    check3_passed = (len(text_mismatches) == 0)
    report["checks"]["question_text_fidelity_check"] = {
        "passed": check3_passed,
        "mismatches_count": len(text_mismatches),
        "mismatches": text_mismatches[:5],
    }
    if not check3_passed:
        report["errors"].append(f"Phát hiện {len(text_mismatches)} câu hỏi sai lệch nội dung so với Phase 9")
    print(f"[{'PASS' if check3_passed else 'FAIL'}] 3. Question Text Fidelity Check: 0 sai lệch nội dung")

    # 4. Ground truth check: relevant_article_ids
    invalid_gt_items = []
    for q in dataset:
        gt_ids = q.get("relevant_article_ids")
        if not isinstance(gt_ids, list) or len(gt_ids) == 0:
            invalid_gt_items.append(q["query_id"])
            continue
        for aid in gt_ids:
            if not isinstance(aid, int) or aid < 0 or aid > 59635:
                invalid_gt_items.append(q["query_id"])
                break

    check4_passed = (len(invalid_gt_items) == 0)
    report["checks"]["ground_truth_check"] = {
        "passed": check4_passed,
        "invalid_gt_queries_count": len(invalid_gt_items),
        "invalid_gt_queries": invalid_gt_items[:5],
    }
    if not check4_passed:
        report["errors"].append(f"Có {len(invalid_gt_items)} câu hỏi ground truth không hợp lệ")
    print(f"[{'PASS' if check4_passed else 'FAIL'}] 4. Ground Truth Check: 100% relevant_article_ids nằm trong [0, 59635]")

    # 5. Difficulty check: single-label vs multi-label
    single_label_count = sum(1 for q in dataset if q.get("difficulty") == "single_label")
    multi_label_count = sum(1 for q in dataset if q.get("difficulty") == "multi_label")
    check5_passed = (single_label_count + multi_label_count == query_count and single_label_count == 1056 and multi_label_count == 345)
    report["checks"]["difficulty_distribution_check"] = {
        "passed": check5_passed,
        "single_label_count": single_label_count,
        "multi_label_count": multi_label_count,
    }
    if not check5_passed:
        report["errors"].append(f"Phân bố độ khó không khớp: single={single_label_count}, multi={multi_label_count}")
    print(
        f"[{'PASS' if check5_passed else 'FAIL'}] 5. Difficulty Distribution Check: "
        f"Single-label = {single_label_count} (75.37%), Multi-label = {multi_label_count} (24.63%)"
    )

    # 6. Question type check
    invalid_qtypes = [q["query_id"] for q in dataset if q.get("question_type") != "legal_retrieval_qa"]
    check6_passed = (len(invalid_qtypes) == 0)
    report["checks"]["question_type_check"] = {
        "passed": check6_passed,
        "invalid_question_type_count": len(invalid_qtypes),
    }
    if not check6_passed:
        report["errors"].append(f"Có {len(invalid_qtypes)} câu hỏi question_type không phải 'legal_retrieval_qa'")
    print(f"[{'PASS' if check6_passed else 'FAIL'}] 6. Question Type Check: 100% legal_retrieval_qa")

    # 7. Split role check
    invalid_splits = [q["query_id"] for q in dataset if q.get("split_role") != "DEVELOPMENT"]
    check7_passed = (len(invalid_splits) == 0)
    report["checks"]["split_role_check"] = {
        "passed": check7_passed,
        "invalid_split_role_count": len(invalid_splits),
    }
    if not check7_passed:
        report["errors"].append(f"Có {len(invalid_splits)} câu hỏi split_role không phải 'DEVELOPMENT'")
    print(f"[{'PASS' if check7_passed else 'FAIL'}] 7. Split Role Check: 100% DEVELOPMENT")

    # 8. Zero Leakage check against FROZEN_TEST
    with open(retrieval_test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_ids = {q["query_id"] for q in test_data}
    test_texts = {q["query_text"].strip().lower() for q in test_data}

    id_overlap = dataset_ids_set.intersection(test_ids)
    text_overlap = {q["query_text"].strip().lower() for q in dataset}.intersection(test_texts)

    check8_passed = (len(id_overlap) == 0 and len(text_overlap) == 0)
    report["checks"]["zero_leakage_check"] = {
        "passed": check8_passed,
        "id_overlap_count": len(id_overlap),
        "text_overlap_count": len(text_overlap),
    }
    if not check8_passed:
        report["errors"].append("Phát hiện rò rỉ dữ liệu với FROZEN_TEST (retrieval_test.json)")
    print(f"[{'PASS' if check8_passed else 'FAIL'}] 8. Zero Leakage Check: 0 ID overlap, 0 text overlap với FROZEN_TEST")

    # 9. Identical Input Contract Check
    # Verify that the input fed to LLM-only is solely query_text without any context/retrieval pollution
    sample_q = dataset[0]
    has_retrieval_artifacts = any(
        k in sample_q for k in ["retrieved_chunks", "context_items", "retrieval_scores", "chunk_ids"]
    )
    check9_passed = not has_retrieval_artifacts
    report["checks"]["identical_input_contract_check"] = {
        "passed": check9_passed,
        "retrieval_contamination_in_dataset": has_retrieval_artifacts,
    }
    if not check9_passed:
        report["errors"].append("Tập dữ liệu chứa trường retrieval không mong muốn")
    print(f"[{'PASS' if check9_passed else 'FAIL'}] 9. Identical Input Contract Check: Không ô nhiễm context/retrieval")

    all_passed = all(
        [
            check1_passed,
            check2_passed,
            check3_passed,
            check4_passed,
            check5_passed,
            check6_passed,
            check7_passed,
            check8_passed,
            check9_passed,
        ]
    )

    report["status"] = "PASSED" if all_passed else "FAILED"
    report["summary"] = {
        "total_queries": query_count,
        "single_label_count": single_label_count,
        "multi_label_count": multi_label_count,
        "all_checks_passed": all_passed,
    }

    atomic_write_json(report_output_path, report)
    print("=" * 80)
    print(f"KẾT QUẢ KIỂM ĐỊNH: {'TẤT CẢ GATES ĐẠT (PASSED)' if all_passed else 'THẤT BẠI (FAILED)'}")
    print(f"Báo cáo chi tiết: {report_output_path}")
    print("=" * 80)

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kiểm định tính toàn vẹn tập dữ liệu đối chuẩn Phase 10"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/benchmarks/phase10_comparison_dataset.json",
        help="Đường dẫn file dataset đối chuẩn",
    )
    args = parser.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.is_absolute():
        ds_path = PROJECT_ROOT / ds_path

    passed = validate_phase10_dataset(dataset_path=ds_path)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
