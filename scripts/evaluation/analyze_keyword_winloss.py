# scripts/evaluation/analyze_keyword_winloss.py
"""Script phân tích đối đầu Win-Loss và Failure Analysis giữa TF-IDF và BM25 Tuned.

Thực hiện:
    1. Kiểm tra tính tương thích và khớp tập hợp query_id giữa 2 mô hình
    2. Phân loại truy vấn theo Win / Tie / Loss dựa trên nDCG@10 (với ngưỡng epsilon)
    3. Nhận diện các nhóm biên: Hard Failures (cả 2 đều R@10=0) và Both Perfect (cả 2 đều R@10=1)
    4. Thống kê phân tầng hiệu năng Single-label vs. Multi-label
    5. Xuất báo cáo JSON và Markdown tổng hợp qua cơ chế ghi nguyên tử (atomic replace)
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set

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

DEFAULT_TFIDF_PATH = (
    PROJECT_ROOT / "results" / "benchmarks" / "tfidf_test_predictions.json"
)
DEFAULT_BM25_PATH = (
    PROJECT_ROOT / "results" / "benchmarks" / "bm25_tuned_test_predictions.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmarks"
DEFAULT_JSON_PATH = DEFAULT_OUTPUT_DIR / "keyword_winloss_analysis.json"
DEFAULT_MD_PATH = (
    PROJECT_ROOT / "docs" / "evaluation" / "keyword_baseline_comparison.md"
)

DEFAULT_EPSILON = 1e-4


def safe_mean(values: List[float]) -> float:
    """Tính trung bình an toàn phòng vệ rỗng."""
    return sum(values) / len(values) if values else 0.0


def compute_group_metrics(
    qids: List[Any],
    tfidf_preds: Dict[Any, Dict[str, Any]],
    bm25_preds: Dict[Any, Dict[str, Any]],
) -> Dict[str, float]:
    """Tính toán metrics trung bình cho một nhóm query ID."""
    if not qids:
        return {
            "tfidf_ndcg10": 0.0,
            "bm25_ndcg10": 0.0,
            "delta_ndcg10": 0.0,
            "tfidf_r10": 0.0,
            "bm25_r10": 0.0,
            "delta_r10": 0.0,
            "tfidf_mrr10": 0.0,
            "bm25_mrr10": 0.0,
            "delta_mrr10": 0.0,
        }

    t_ndcg = safe_mean([tfidf_preds[q]["metrics"].get("nDCG@10", 0.0) for q in qids])
    b_ndcg = safe_mean([bm25_preds[q]["metrics"].get("nDCG@10", 0.0) for q in qids])
    t_r10 = safe_mean([tfidf_preds[q]["metrics"].get("R@10", 0.0) for q in qids])
    b_r10 = safe_mean([bm25_preds[q]["metrics"].get("R@10", 0.0) for q in qids])
    t_mrr = safe_mean([tfidf_preds[q]["metrics"].get("MRR@10", 0.0) for q in qids])
    b_mrr = safe_mean([bm25_preds[q]["metrics"].get("MRR@10", 0.0) for q in qids])

    return {
        "tfidf_ndcg10": round(t_ndcg, 4),
        "bm25_ndcg10": round(b_ndcg, 4),
        "delta_ndcg10": round(b_ndcg - t_ndcg, 4),
        "tfidf_r10": round(t_r10, 4),
        "bm25_r10": round(b_r10, 4),
        "delta_r10": round(b_r10 - t_r10, 4),
        "tfidf_mrr10": round(t_mrr, 4),
        "bm25_mrr10": round(b_mrr, 4),
        "delta_mrr10": round(b_mrr - t_mrr, 4),
    }


def analyze_keyword_winloss(
    tfidf_pred_path: Path = DEFAULT_TFIDF_PATH,
    bm25_pred_path: Path = DEFAULT_BM25_PATH,
    json_output_path: Path = DEFAULT_JSON_PATH,
    md_output_path: Path = DEFAULT_MD_PATH,
    epsilon: float = DEFAULT_EPSILON,
) -> int:
    """Thực thi phân tích đối đầu Win-Loss và xuất tài liệu nghiệm thu."""
    print("=" * 70)
    print("BƯỚC 5.11: PHÂN TÍCH ĐỐI ĐẦU WIN-LOSS & FAILURE ANALYSIS")
    print("=" * 70)

    if not tfidf_pred_path.exists():
        print(f"[FAIL] Không tìm thấy dự đoán TF-IDF: {tfidf_pred_path}")
        return 1
    if not bm25_pred_path.exists():
        print(f"[FAIL] Không tìm thấy dự đoán BM25 Tuned: {bm25_pred_path}")
        return 1

    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Nạp dữ liệu dự đoán chi tiết
    try:
        with open(tfidf_pred_path, "r", encoding="utf-8") as f:
            tfidf_raw = json.load(f)
        with open(bm25_pred_path, "r", encoding="utf-8") as f:
            bm25_raw = json.load(f)
    except Exception as e:
        print(f"[FAIL] Lỗi khi nạp file predictions: {e}")
        return 1

    tfidf_preds = {item["query_id"]: item for item in tfidf_raw if "query_id" in item}
    bm25_preds = {item["query_id"]: item for item in bm25_raw if "query_id" in item}

    # Kiểm tra tính khớp 100% của tập Query IDs
    tfidf_qids = set(tfidf_preds.keys())
    bm25_qids = set(bm25_preds.keys())

    if tfidf_qids != bm25_qids:
        missing_in_bm25 = tfidf_qids - bm25_qids
        missing_in_tfidf = bm25_qids - tfidf_qids
        print(
            f"[FAIL] Tập hợp query_id không khớp giữa 2 mô hình!\n"
            f"       Thiếu trong BM25: {len(missing_in_bm25)}, Thiếu trong TF-IDF: {len(missing_in_tfidf)}"
        )
        return 1

    total_queries = len(tfidf_preds)
    print(f"1. Đã nạp và đối chiếu khớp {total_queries:,} queries từ cả hai mô hình.")
    if total_queries == 0:
        print("[FAIL] Danh sách truy vấn rỗng.")
        return 1

    # 2. Phân loại Win / Tie / Loss
    bm25_wins: List[Dict[str, Any]] = []
    ties: List[Dict[str, Any]] = []
    bm25_losses: List[Dict[str, Any]] = []

    hard_failures: List[Dict[str, Any]] = []
    both_perfect: List[Dict[str, Any]] = []

    single_label_qids: List[Any] = []
    multi_label_qids: List[Any] = []

    for qid, t_item in tfidf_preds.items():
        b_item = bm25_preds[qid]
        t_ndcg = t_item["metrics"].get("nDCG@10", 0.0)
        b_ndcg = b_item["metrics"].get("nDCG@10", 0.0)

        t_r10 = t_item["metrics"].get("R@10", 0.0)
        b_r10 = b_item["metrics"].get("R@10", 0.0)

        diff = b_ndcg - t_ndcg

        query_info = {
            "query_id": qid,
            "query_text": t_item.get("query_text", ""),
            "ground_truth": t_item.get("ground_truth", []),
            "is_multi_label": t_item.get("is_multi_label", False),
            "tfidf_ndcg10": t_ndcg,
            "bm25_ndcg10": b_ndcg,
            "delta_ndcg10": round(diff, 4),
            "tfidf_r10": t_r10,
            "bm25_r10": b_r10,
        }

        if t_item.get("is_multi_label", False):
            multi_label_qids.append(qid)
        else:
            single_label_qids.append(qid)

        if diff > epsilon:
            bm25_wins.append(query_info)
        elif diff < -epsilon:
            bm25_losses.append(query_info)
        else:
            ties.append(query_info)

        if t_r10 == 0.0 and b_r10 == 0.0:
            hard_failures.append(query_info)
        elif t_r10 >= 1.0 and b_r10 >= 1.0:
            both_perfect.append(query_info)

    win_rate = (len(bm25_wins) / total_queries) * 100
    tie_rate = (len(ties) / total_queries) * 100
    loss_rate = (len(bm25_losses) / total_queries) * 100
    hard_fail_rate = (len(hard_failures) / total_queries) * 100
    both_perf_rate = (len(both_perfect) / total_queries) * 100

    # 3. Phân tích phân tầng
    single_stats = compute_group_metrics(single_label_qids, tfidf_preds, bm25_preds)
    multi_stats = compute_group_metrics(multi_label_qids, tfidf_preds, bm25_preds)
    all_stats = compute_group_metrics(list(tfidf_preds.keys()), tfidf_preds, bm25_preds)

    # 4. In bảng tóm tắt
    print("\n" + "=" * 70)
    print("PHÂN PHỐI WIN / TIE / LOSS (THEO nDCG@10)")
    print("=" * 70)
    print(f"BM25 Wins   (Cải thiện)     : {len(bm25_wins):>3} queries ({win_rate:.2f}%)")
    print(f"Ties        (Ngang bằng)    : {len(ties):>3} queries ({tie_rate:.2f}%)")
    print(f"BM25 Losses (TF-IDF tốt hơn): {len(bm25_losses):>3} queries ({loss_rate:.2f}%)")
    print("-" * 70)
    print(f"Both Perfect  (Cả hai R@10 = 1.0): {len(both_perfect):>3} queries ({both_perf_rate:.2f}%)")
    print(f"Hard Failures (Cả hai R@10 = 0.0): {len(hard_failures):>3} queries ({hard_fail_rate:.2f}%)")

    print("\n" + "=" * 70)
    print("HIỆU NĂNG PHÂN TẦNG: SINGLE-LABEL VS. MULTI-LABEL")
    print("=" * 70)
    print(f"{'Nhóm':<14} | {'TF nDCG@10':<11} | {'BM25 nDCG@10':<13} | {'TF R@10':<9} | {'BM25 R@10':<10}")
    print("-" * 70)
    print(f"Single ({len(single_label_qids):<3})   | {single_stats['tfidf_ndcg10']:<11.4f} | {single_stats['bm25_ndcg10']:<13.4f} | {single_stats['tfidf_r10']:<9.4f} | {single_stats['bm25_r10']:<10.4f}")
    print(f"Multi  ({len(multi_label_qids):<3})   | {multi_stats['tfidf_ndcg10']:<11.4f} | {multi_stats['bm25_ndcg10']:<13.4f} | {multi_stats['tfidf_r10']:<9.4f} | {multi_stats['bm25_r10']:<10.4f}")
    print("=" * 70)

    # 5. Lưu JSON an toàn
    analysis_payload = {
        "total_queries": total_queries,
        "win_loss_summary": {
            "bm25_wins": len(bm25_wins),
            "ties": len(ties),
            "bm25_losses": len(bm25_losses),
            "hard_failures_both_zero": len(hard_failures),
            "both_perfect_recall": len(both_perfect),
        },
        "overall_metrics": all_stats,
        "stratified_metrics": {
            "single_label": single_stats,
            "multi_label": multi_stats,
        },
        "bm25_losses_cases": bm25_losses,
        "hard_failures_cases": hard_failures,
    }

    tmp_json = json_output_path.with_suffix(".json.tmp")
    tmp_md = md_output_path.with_suffix(".md.tmp")

    try:
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(analysis_payload, f, ensure_ascii=False, indent=2)

        # 6. Viết báo cáo Markdown an toàn
        with open(tmp_md, "w", encoding="utf-8") as f:
            f.write("# Báo Cáo Đối Chuẩn: TF-IDF vs. BM25 Tuned (Official Keyword Baseline)\n\n")
            f.write(
                f"Tài liệu này ghi lại chi tiết các phân tích so sánh đối chuẩn giữa mô hình TF-IDF "
                f"và BM25 Tuned trên {total_queries:,} test queries thuộc `{tfidf_pred_path.name}`.\n\n"
            )

            f.write("## 1. Phân Phối Win / Tie / Loss\n\n")
            f.write(f"- **BM25 Wins:** {len(bm25_wins)} queries ({win_rate:.2f}%)\n")
            f.write(f"- **Ties:** {len(ties)} queries ({tie_rate:.2f}%)\n")
            f.write(f"- **BM25 Losses:** {len(bm25_losses)} queries ({loss_rate:.2f}%)\n")
            f.write(f"- **Hard Failures ($R@10=0$ cả hai):** {len(hard_failures)} queries ({hard_fail_rate:.2f}%)\n\n")

            f.write("## 2. Hiệu Năng Phân Tầng (Single-label vs. Multi-label)\n\n")
            f.write("| Phân nhóm | Số queries | TF-IDF nDCG@10 | BM25 nDCG@10 | Delta nDCG@10 | TF-IDF R@10 | BM25 R@10 | Delta R@10 |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            f.write(f"| **Single-label** | {len(single_label_qids)} | {single_stats['tfidf_ndcg10']:.4f} | {single_stats['bm25_ndcg10']:.4f} | {single_stats['delta_ndcg10']:+.4f} | {single_stats['tfidf_r10']:.4f} | {single_stats['bm25_r10']:.4f} | {single_stats['delta_r10']:+.4f} |\n")
            f.write(f"| **Multi-label** | {len(multi_label_qids)} | {multi_stats['tfidf_ndcg10']:.4f} | {multi_stats['bm25_ndcg10']:.4f} | {multi_stats['delta_ndcg10']:+.4f} | {multi_stats['tfidf_r10']:.4f} | {multi_stats['bm25_r10']:.4f} | {multi_stats['delta_r10']:+.4f} |\n\n")

            f.write("## 3. Phân Tích Trường Hợp BM25 Losses (TF-IDF thắng)\n\n")
            f.write(f"Có tổng cộng **{len(bm25_losses)}** truy vấn mà TF-IDF đạt nDCG@10 cao hơn BM25. Dưới đây là các ví dụ tiêu biểu:\n\n")
            for ex in bm25_losses[:5]:
                f.write(f"- **Query {ex['query_id']}:** \"{ex['query_text']}\"\n")
                f.write(f"  - Ground Truth: `{ex['ground_truth']}`\n")
                f.write(f"  - TF-IDF nDCG@10: `{ex['tfidf_ndcg10']:.4f}` | BM25 nDCG@10: `{ex['bm25_ndcg10']:.4f}` (Delta: `{ex['delta_ndcg10']:+.4f}`)\n")
                f.write(f"  - TF-IDF R@10: `{ex['tfidf_r10']:.4f}` | BM25 R@10: `{ex['bm25_r10']:.4f}`\n\n")

            f.write("## 4. Phân Tích Hard Failures (Giới Hạn Vật Lý Của Keyword Search)\n\n")
            f.write(f"Có **{len(hard_failures)}** queries mà cả TF-IDF và BM25 đều không thể tìm thấy bất kỳ điều luật liên quan nào trong Top 10 ($R@10 = 0$).\n\n")
            f.write("Các đặc trưng cốt lõi của nhóm câu hỏi này:\n")
            f.write("1. **Hiện tượng diễn giải đồng nghĩa (Paraphrasing):** Người dùng dùng từ thông dụng đời thường thay cho thuật ngữ pháp lý.\n")
            f.write("2. **Truy vấn trừu tượng / suy luận (Abstract / Multi-hop):** Câu hỏi mô tả hành vi vi phạm mà không gọi tên chính xác điều luật.\n")
            f.write("3. **Khoảng cách từ vựng (Vocabulary Mismatch):** Query và tài liệu liên quan có mức độ giao thoa từ vựng thấp.\n\n")
            for ex in hard_failures[:5]:
                f.write(f"- **Query {ex['query_id']}:** \"{ex['query_text']}\"\n")
                f.write(f"  - Ground Truth: `{ex['ground_truth']}`\n")
                f.write(f"  - Multi-label: `{ex['is_multi_label']}`\n\n")

            f.write("## 5. Kết Luận Kỹ Thuật Chuyển Tiếp Sang Phase 6\n\n")
            f.write(
                f"- BM25 Tuned cho thấy hiệu năng tổng thể tốt hơn TF-IDF trên các metric đối chuẩn chính. "
                f"BM25 sử dụng cơ chế Term Frequency Saturation và Document Length Normalization, "
                f"với nDCG@10 = {all_stats['bm25_ndcg10']:.4f} và "
                f"Recall@10 = {all_stats['bm25_r10']*100:.2f}%.\n"
            )
            f.write(
                f"- Tuy nhiên, rào cản lớn nhất của phương pháp từ khóa là **{len(hard_failures)} queries thất bại hoàn toàn** "
                f"({hard_fail_rate:.2f}%). Đây là giới hạn không thể khắc phục nếu chỉ dùng lexical matching.\n"
            )
            f.write(
                "- **Nhiệm vụ Phase 6 (Dense Retrieval):** Triển khai mô hình Bi-Encoder embedding ngữ nghĩa tiếng Việt "
                "nhằm thu hẹp khoảng cách từ vựng và xử lý triệt để nhóm Hard Failures này.\n"
            )

        tmp_json.replace(json_output_path)
        tmp_md.replace(md_output_path)
    finally:
        for tmp_file in [tmp_json, tmp_md]:
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)

    print("Artifacts đã lưu thành công:")
    print(f"  - Analysis JSON : {json_output_path}")
    print(f"  - Report MD     : {md_output_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phân tích đối đầu Win-Loss giữa TF-IDF và BM25 Tuned."
    )
    parser.add_argument(
        "--tfidf-preds",
        type=Path,
        default=DEFAULT_TFIDF_PATH,
        help="Đường dẫn file tfidf_test_predictions.json.",
    )
    parser.add_argument(
        "--bm25-preds",
        type=Path,
        default=DEFAULT_BM25_PATH,
        help="Đường dẫn file bm25_tuned_test_predictions.json.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help="Đường dẫn lưu file kết quả phân tích JSON.",
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=DEFAULT_MD_PATH,
        help="Đường dẫn lưu file markdown đối chuẩn.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help="Ngưỡng chênh lệch nDCG@10 để tính hòa (Ties) (mặc định: 1e-4).",
    )
    args = parser.parse_args()

    exit_code = analyze_keyword_winloss(
        tfidf_pred_path=args.tfidf_preds,
        bm25_pred_path=args.bm25_preds,
        json_output_path=args.json_output,
        md_output_path=args.md_output,
        epsilon=args.epsilon,
    )
    sys.exit(exit_code)