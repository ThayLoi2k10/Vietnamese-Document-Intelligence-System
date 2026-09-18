# scripts/evaluation/analyze_dense_vs_bm25.py
"""Script phân tích sai số chuyên sâu và kiểm tra khả năng giải cứu (Rescue Analysis).

Nghiên cứu câu hỏi cốt lõi của Phase 6:
    "Mô hình Dense Retrieval có thực sự giải cứu (rescue) được các câu hỏi
     mà BM25 gặp thất bại nặng nề (Hard Failures: R@10 = 0) do rào cản từ vựng hay không?"

Thực hiện:
    1. Đọc kết quả dự đoán chi tiết của BM25 Tuned và Dense BGE-M3.
    2. Xác định tập hard-failure của BM25 (R@10 == 0) động theo kết quả thực tế.
    3. Thiết lập ma trận 4 kịch bản (Four-way Matrix):
       - Cả hai cùng thành công (Both Succeed)
       - BM25 thành công nhưng Dense thất bại (BM25 Only)
       - Dense giải cứu thành công BM25 (Dense Rescue)
       - Cả hai cùng thất bại (Both Fail)
    4. Phân tích định lượng và phân phối thứ hạng, điểm số của các trường hợp giải cứu.
    5. Lấy mẫu ngẫu nhiên tất định (Deterministic Sampling qua MD5 hash) tối đa 20 ca giải cứu
       và 20 ca cùng thất bại để phục vụ phân tích chất lượng (Qualitative Error Analysis).
    6. Xuất artifact dense_vs_bm25_hard_failure_analysis.json an toàn.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional
import numpy as np

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

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmarks"
DEFAULT_BM25_PREDS = DEFAULT_OUTPUT_DIR / "bm25_tuned_test_predictions.json"
DEFAULT_DENSE_PREDS = DEFAULT_OUTPUT_DIR / "dense_test_predictions.json"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "dense_vs_bm25_hard_failure_analysis.json"


def deterministic_sample(items: List[Dict[str, Any]], sample_size: int = 20) -> List[Dict[str, Any]]:
    """Chọn mẫu tất định dựa trên MD5 hash của query_id để tránh cherry-picking."""
    if len(items) <= sample_size:
        return items

    def _hash_key(item: Dict[str, Any]) -> str:
        qid = str(item.get("query_id", ""))
        return hashlib.md5(qid.encode("utf-8")).hexdigest()

    sorted_items = sorted(items, key=_hash_key)
    return sorted_items[:sample_size]


def atomic_write_json(file_path: Path, data: Any) -> None:
    """Ghi an toàn file JSON."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=file_path.parent, delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp_name = tmp.name
    os.replace(tmp_name, file_path)


def analyze_dense_vs_bm25(
    bm25_preds_path: Path = DEFAULT_BM25_PREDS,
    dense_preds_path: Path = DEFAULT_DENSE_PREDS,
    output_path: Path = DEFAULT_REPORT_PATH,
) -> int:
    print("=" * 70)
    print("PHASE 6: PHÂN TÍCH HARD-FAILURE VÀ ĐÁNH GIÁ KHẢ NĂNG GIẢI CỨU CỦA DENSE")
    print("=" * 70)

    if not bm25_preds_path.exists():
        print(f"[FAIL] Không tìm thấy dự đoán BM25 tại: {bm25_preds_path}")
        return 1
    if not dense_preds_path.exists():
        print(f"[FAIL] Không tìm thấy dự đoán Dense tại: {dense_preds_path}")
        return 1

    with open(bm25_preds_path, "r", encoding="utf-8") as f:
        bm25_preds = {item["query_id"]: item for item in json.load(f)}
    with open(dense_preds_path, "r", encoding="utf-8") as f:
        dense_preds = {item["query_id"]: item for item in json.load(f)}

    common_qids = sorted(list(set(bm25_preds.keys()) & set(dense_preds.keys())))
    num_total = len(common_qids)
    print(f"Tổng số query chung phân tích: {num_total:,}")
    if num_total == 0:
        print("[FAIL] Không có query chung nào giữa hai tập dự đoán.")
        return 1

    # 1. Phân loại 4 nhóm
    both_succeed = []
    bm25_only = []
    dense_rescue = []
    both_fail = []

    for qid in common_qids:
        b_item = bm25_preds[qid]
        d_item = dense_preds[qid]

        b_r10 = b_item["metrics"].get("R@10", 0.0)
        d_r10 = d_item["metrics"].get("R@10", 0.0)

        record = {
            "query_id": qid,
            "query_text": b_item["query_text"],
            "ground_truth": b_item["ground_truth"],
            "is_multi_label": b_item.get("is_multi_label", False),
            "bm25_metrics": b_item["metrics"],
            "dense_metrics": d_item["metrics"],
            "bm25_top_10": b_item.get("top_10_retrieved", []),
            "dense_top_10": d_item.get("top_10_retrieved", []),
        }

        # Tìm thứ hạng xuất hiện đầu tiên của tài liệu liên quan trong Dense
        first_rel_rank_dense = None
        for res in d_item.get("top_10_retrieved", []):
            if res.get("is_relevant"):
                first_rel_rank_dense = res.get("rank")
                break
        record["dense_first_relevant_rank"] = first_rel_rank_dense

        if b_r10 > 0 and d_r10 > 0:
            both_succeed.append(record)
        elif b_r10 > 0 and d_r10 == 0:
            bm25_only.append(record)
        elif b_r10 == 0 and d_r10 > 0:
            dense_rescue.append(record)
        else:
            both_fail.append(record)

    total_bm25_failures = len(dense_rescue) + len(both_fail)
    total_bm25_success = len(both_succeed) + len(bm25_only)

    rescue_rate_among_bm25_failures = (
        len(dense_rescue) / total_bm25_failures if total_bm25_failures > 0 else 0.0
    )

    print("\n" + "=" * 70)
    print("MA TRẬN HIỆU QUẢ 4 KỊCH BẢN (FOUR-WAY OUTCOME MATRIX):")
    print("=" * 70)
    print(f"1. Cả hai cùng thành công (Both Succeed):        {len(both_succeed):4d} queries ({len(both_succeed)/num_total:.2%})")
    print(f"2. BM25 thành công, Dense thất bại (BM25 Only):  {len(bm25_only):4d} queries ({len(bm25_only)/num_total:.2%})")
    print(f"3. DENSE GIẢI CỨU THÀNH CÔNG (Dense Rescue):     {len(dense_rescue):4d} queries ({len(dense_rescue)/num_total:.2%})")
    print(f"4. Cả hai cùng thất bại (Both Fail):             {len(both_fail):4d} queries ({len(both_fail)/num_total:.2%})")
    print("-" * 70)
    print(f"Tổng số BM25 Hard Failures (R@10=0):             {total_bm25_failures:4d} queries ({total_bm25_failures/num_total:.2%})")
    print(f"Tỷ lệ Dense giải cứu thành công:                 {rescue_rate_among_bm25_failures:.2%} "
          f"({len(dense_rescue)}/{total_bm25_failures} ca)")
    print("=" * 70)

    # 2. Thống kê chi tiết các ca giải cứu
    rescue_ranks = [r["dense_first_relevant_rank"] for r in dense_rescue if r["dense_first_relevant_rank"] is not None]
    rank_dist = {f"Rank {k}": rescue_ranks.count(k) for k in range(1, 11)}

    print("\nPhân bổ thứ hạng tài liệu liên quan trong các ca Dense giải cứu:")
    for k in range(1, 11):
        c = rank_dist.get(f"Rank {k}", 0)
        if c > 0:
            print(f"   * Rank {k:2d}: {c:3d} queries ({c/len(dense_rescue):.2%})")

    # 3. Lấy mẫu tất định 20 ca giải cứu và 20 ca cùng thất bại
    sampled_rescues = deterministic_sample(dense_rescue, sample_size=20)
    sampled_both_fail = deterministic_sample(both_fail, sample_size=20)

    report_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries_analyzed": num_total,
        "bm25_predictions_source": str(bm25_preds_path.name),
        "dense_predictions_source": str(dense_preds_path.name),
        "four_way_matrix": {
            "both_succeed_count": len(both_succeed),
            "both_succeed_rate": round(len(both_succeed) / num_total, 4),
            "bm25_only_count": len(bm25_only),
            "bm25_only_rate": round(len(bm25_only) / num_total, 4),
            "dense_rescue_count": len(dense_rescue),
            "dense_rescue_rate_overall": round(len(dense_rescue) / num_total, 4),
            "both_fail_count": len(both_fail),
            "both_fail_rate": round(len(both_fail) / num_total, 4),
        },
        "rescue_analysis": {
            "total_bm25_hard_failures": total_bm25_failures,
            "rescue_count": len(dense_rescue),
            "rescue_rate_among_failures": round(rescue_rate_among_bm25_failures, 4),
            "first_relevant_rank_distribution": rank_dist,
        },
        "deterministic_samples": {
            "rescue_sample_count": len(sampled_rescues),
            "rescue_samples": sampled_rescues,
            "both_fail_sample_count": len(sampled_both_fail),
            "both_fail_samples": sampled_both_fail,
        },
    }

    atomic_write_json(output_path, report_payload)
    print(f"\nĐã ghi báo cáo phân tích chi tiết tại: {output_path}")
    print("=" * 70)
    print("[PASS] HOÀN TẤT PHÂN TÍCH SAI SỐ VÀ GIẢI CỨU THÀNH CÔNG")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phân tích lỗi và so sánh Dense vs BM25")
    parser.add_argument("--bm25-preds", type=Path, default=DEFAULT_BM25_PREDS)
    parser.add_argument("--dense-preds", type=Path, default=DEFAULT_DENSE_PREDS)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_REPORT_PATH)

    args = parser.parse_args()
    sys.exit(
        analyze_dense_vs_bm25(
            bm25_preds_path=args.bm25_preds,
            dense_preds_path=args.dense_preds,
            output_path=args.output_path,
        )
    )
