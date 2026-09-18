# scripts/evaluation/analyze_tfidf_errors.py
"""Script phân tích lỗi chuyên sâu (Error Analysis) cho mô hình TF-IDF Baseline.

Thực hiện:
    1. Validate dải giá trị của tham số --threshold [0, 1]
    2. Nạp log dự đoán chi tiết (tfidf_test_predictions.json) và bảng corpus
    3. Guard chống tập dữ liệu rỗng và xử lý an toàn phép chia tỷ lệ
    4. Tokenize làm sạch dấu câu để đo lường độ trùng lặp từ vựng chính xác
    5. Bóc tách 2 failure modes: Lexical Chasm vs. Lexical Competition / Ranking Failure Candidates
    6. Đánh giá khoảng cách hiệu năng giữa Single-label và Multi-label
    7. Xuất báo cáo Markdown chi tiết tại docs/evaluation/ qua atomic replace
"""

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Set
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

# Thêm PROJECT_ROOT vào sys.path để import được src
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PREDICTIONS_PATH = (
    PROJECT_ROOT / "results" / "benchmarks" / "tfidf_test_predictions.json"
)
DEFAULT_CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_retrieval"
    / "corpus_articles_preprocessed.parquet"
)
DEFAULT_OUTPUT_REPORT_PATH = (
    PROJECT_ROOT / "docs" / "evaluation" / "tfidf_error_analysis.md"
)

# Regex làm sạch dấu câu cơ bản giữ lại chữ và số tiếng Việt
CLEAN_PUNCT_REGEX = re.compile(r"[^\w\s]", re.UNICODE)


def tokenize_clean(text: Any) -> Set[str]:
    """Tách từ, chuyển chữ thường và loại bỏ dấu câu dính liền."""
    if not isinstance(text, str) or not text.strip():
        return set()
    cleaned = CLEAN_PUNCT_REGEX.sub(" ", text.lower())
    return set(cleaned.split())


def mean_metric(items: List[Dict[str, Any]], key: str) -> float:
    """Tính giá trị trung bình an toàn của metric trong danh sách."""
    if not items:
        return 0.0
    return sum(item["metrics"].get(key, 0.0) for item in items) / len(items)


def analyze_tfidf_errors(
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    output_report_path: Path = DEFAULT_OUTPUT_REPORT_PATH,
    overlap_threshold: float = 0.35,
) -> int:
    """Thực thi phân tích lỗi chuyên sâu cho kết quả TF-IDF."""
    print("=" * 65)
    print("BƯỚC 5.6: PHÂN TÍCH LỖI CHUYÊN SÂU TF-IDF (ERROR ANALYSIS)")
    print("=" * 65)

    # 1. Guard threshold CLI
    if not (0.0 <= overlap_threshold <= 1.0):
        print(f"[FAIL] --threshold phải nằm trong khoảng [0, 1] (nhận giá trị: {overlap_threshold}).")
        return 1

    if not predictions_path.exists():
        print(f"[FAIL] Không tìm thấy predictions log: {predictions_path}")
        return 1
    if not corpus_path.exists():
        print(f"[FAIL] Không tìm thấy corpus: {corpus_path}")
        return 1

    output_report_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Nạp predictions log
    print(f"1. Đọc predictions log: {predictions_path.name}...")
    try:
        with open(predictions_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)
    except Exception as e:
        print(f"[FAIL] Lỗi khi đọc file predictions: {e}")
        return 1

    total_q = len(predictions)
    print(f"   -> Đã nạp {total_q:,} predictions.")
    if total_q == 0:
        print("[FAIL] File predictions log rỗng bản ghi.")
        return 1

    # 3. Nạp corpus map {article_id: text}
    print(f"2. Nạp corpus để đối chiếu từ vựng...")
    try:
        corpus_table = pq.read_table(
            corpus_path, columns=["global_article_id", "text"]
        )
        corpus_map = dict(
            zip(
                corpus_table["global_article_id"].to_pylist(),
                corpus_table["text"].to_pylist(),
            )
        )
    except Exception as e:
        print(f"[FAIL] Lỗi khi đọc corpus: {e}")
        return 1

    print(f"   -> Đã lập bản đồ {len(corpus_map):,} điều luật.")

    # 4. Phân nhóm theo kết quả Top 10
    complete_failures: List[Dict[str, Any]] = []
    partial_successes: List[Dict[str, Any]] = []
    complete_successes: List[Dict[str, Any]] = []

    single_label_preds: List[Dict[str, Any]] = []
    multi_label_preds: List[Dict[str, Any]] = []

    for pred in predictions:
        r10 = pred["metrics"].get("R@10", 0.0)
        if r10 == 0.0:
            complete_failures.append(pred)
        elif r10 >= 1.0:
            complete_successes.append(pred)
        else:
            partial_successes.append(pred)

        if pred.get("is_multi_label", False):
            multi_label_preds.append(pred)
        else:
            single_label_preds.append(pred)

    fail_rate = (len(complete_failures) / total_q) * 100
    comp_rate = (len(complete_successes) / total_q) * 100
    part_rate = (len(partial_successes) / total_q) * 100

    print("\n--- PHÂN PHỐI HIỆU QUẢ TẠI TOP-10 ---")
    print(f"Complete Success (R@10 = 1.0) : {len(complete_successes):>3} queries ({comp_rate:.2f}%)")
    print(f"Partial Success  (0 < R@10 < 1): {len(partial_successes):>3} queries ({part_rate:.2f}%)")
    print(f"Complete Failure (R@10 = 0.0) : {len(complete_failures):>3} queries ({fail_rate:.2f}%)")

    # 5. So sánh Single-label vs Multi-label
    print("\n--- SINGLE-LABEL VS MULTI-LABEL ---")
    print(f"{'Nhóm':<15} | {'Số lượng':<8} | {'MRR@10':<8} | {'R@10':<8} | {'MAP@10':<8} | {'nDCG@10':<8}")
    print("-" * 65)
    for name, group in [("Single-label", single_label_preds), ("Multi-label", multi_label_preds)]:
        print(
            f"{name:<15} | {len(group):<8} | "
            f"{mean_metric(group, 'MRR@10'):<8.4f} | "
            f"{mean_metric(group, 'R@10'):<8.4f} | "
            f"{mean_metric(group, 'MAP@10'):<8.4f} | "
            f"{mean_metric(group, 'nDCG@10'):<8.4f}"
        )

    # 6. Phân tích nguyên nhân nhóm Complete Failure (R@10 = 0)
    print("\n--- PHÂN TÍCH NHÓM COMPLETE FAILURE (R@10 = 0) ---")
    lexical_chasm_cases: List[Dict[str, Any]] = []
    competition_cases: List[Dict[str, Any]] = []

    for pred in complete_failures:
        q_tokens = tokenize_clean(pred.get("query_text", ""))
        gt_tokens: Set[str] = set()

        for aid in pred.get("ground_truth", []):
            if aid in corpus_map:
                gt_tokens.update(tokenize_clean(corpus_map[aid]))

        overlap = q_tokens.intersection(gt_tokens)
        overlap_ratio = len(overlap) / len(q_tokens) if q_tokens else 0.0

        top_10 = pred.get("top_10_retrieved", [])
        top_1_score = top_10[0]["score"] if top_10 else 0.0
        top_1_id = top_10[0]["article_id"] if top_10 else None

        info = {
            "query_id": pred.get("query_id"),
            "query_text": pred.get("query_text"),
            "ground_truth": pred.get("ground_truth"),
            "overlap_tokens": sorted(list(overlap)),
            "overlap_ratio": round(overlap_ratio, 4),
            "top_1_score": top_1_score,
            "top_1_id": top_1_id,
        }

        if overlap_ratio <= overlap_threshold:
            lexical_chasm_cases.append(info)
        else:
            competition_cases.append(info)

    num_failures = len(complete_failures)
    chasm_pct = (len(lexical_chasm_cases) / num_failures * 100) if num_failures > 0 else 0.0
    comp_pct = (len(competition_cases) / num_failures * 100) if num_failures > 0 else 0.0

    print(f"Tổng số queries thất bại hoàn toàn : {num_failures}")
    print(f"  * Do Lexical Chasm (overlap <= {overlap_threshold*100:.0f}%): {len(lexical_chasm_cases)} ({chasm_pct:.1f}%)")
    print(f"  * Do Lexical Competition / Ranking Failure Candidates (overlap > {overlap_threshold*100:.0f}%): {len(competition_cases)} ({comp_pct:.1f}%)")

    # 7. Tạo tài liệu báo cáo Markdown an toàn
    tmp_report = output_report_path.with_suffix(".md.tmp")
    try:
        with open(tmp_report, "w", encoding="utf-8") as f:
            f.write("# Phân Tích Lỗi Thực Nghiệm: TF-IDF Baseline\n\n")
            f.write(
                "Tài liệu này ghi lại chi tiết các failure modes của mô hình `TF-IDF` "
                f"trên {total_q:,} queries thuộc `{predictions_path.name}`.\n\n"
            )

            f.write("## 1. Bức Tranh Tổng Thể Tại Top-10\n\n")
            f.write(f"- **Tổng số truy vấn test:** {total_q:,}\n")
            f.write(f"- **Thành công trọn vẹn ($R@10 = 1.0$):** {len(complete_successes):,} queries ({comp_rate:.2f}%)\n")
            f.write(f"- **Thành công một phần ($0 < R@10 < 1.0$):** {len(partial_successes):,} queries ({part_rate:.2f}%)\n")
            f.write(f"- **Thất bại hoàn toàn ($R@10 = 0.0$):** {len(complete_failures):,} queries ({fail_rate:.2f}%)\n\n")

            f.write("## 2. Khoảng Cách Hiệu Năng Giữa Single-label và Multi-label\n\n")
            f.write("| Nhóm truy vấn | Số lượng | MRR@10 | Recall@10 | MAP@10 | nDCG@10 |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
            f.write(
                f"| **Single-label** | {len(single_label_preds):,} | "
                f"{mean_metric(single_label_preds, 'MRR@10'):.4f} | "
                f"{mean_metric(single_label_preds, 'R@10'):.4f} | "
                f"{mean_metric(single_label_preds, 'MAP@10'):.4f} | "
                f"{mean_metric(single_label_preds, 'nDCG@10'):.4f} |\n"
            )
            f.write(
                f"| **Multi-label** | {len(multi_label_preds):,} | "
                f"{mean_metric(multi_label_preds, 'MRR@10'):.4f} | "
                f"{mean_metric(multi_label_preds, 'R@10'):.4f} | "
                f"{mean_metric(multi_label_preds, 'MAP@10'):.4f} | "
                f"{mean_metric(multi_label_preds, 'nDCG@10'):.4f} |\n\n"
            )
            f.write(
                "> **Nhận xét:** Trên các truy vấn Multi-label, Recall@10 sụt giảm rõ rệt. "
                "Cơ chế so khớp từ khóa độc lập không thể bắt trọn vẹn các điều luật bổ trợ hoặc quy định viện dẫn.\n\n"
            )

            f.write("## 3. Phân Loại Nguyên Nhân Thất Bại (Error Taxonomy)\n\n")
            f.write(f"Trong **{num_failures}** queries thất bại hoàn toàn ($R@10 = 0$):\n\n")

            f.write(
                f"### A. Lexical Chasm / Vocabulary Mismatch ({len(lexical_chasm_cases)} queries - {chasm_pct:.1f}%)\n\n"
            )
            f.write("- **Bản chất:** Từ vựng giữa câu hỏi người dùng và văn bản luật không giao thoa đủ lớn.\n")
            f.write("- **Ví dụ tiêu biểu:**\n\n")
            for ex in lexical_chasm_cases[:5]:
                f.write(f"  * **Query ID {ex['query_id']}:** \"{ex['query_text']}\"\n")
                f.write(f"    - Ground truth IDs: `{ex['ground_truth']}`\n")
                f.write(f"    - Từ khóa trùng: `{ex['overlap_tokens']}` (Tỷ lệ trùng: {ex['overlap_ratio']*100:.1f}%)\n")
                f.write(f"    - Top 1 retrieved ID: `{ex['top_1_id']}` (Score: {ex['top_1_score']})\n\n")

            f.write(
                f"### B. Lexical Competition / Ranking Failure Candidates ({len(competition_cases)} queries - {comp_pct:.1f}%)\n\n"
            )
            f.write("- **Bản chất:** Query có từ khóa giao với ground-truth nhưng các tài liệu khác có đặc trưng từ vựng tương đồng hơn theo TF-IDF, khiến ground-truth bị xếp hạng ngoài Top-10.\n")
            f.write("- **Ví dụ tiêu biểu:**\n\n")
            for ex in competition_cases[:5]:
                f.write(f"  * **Query ID {ex['query_id']}:** \"{ex['query_text']}\"\n")
                f.write(f"    - Ground truth IDs: `{ex['ground_truth']}`\n")
                f.write(f"    - Từ khóa trùng: `{ex['overlap_tokens']}`\n")
                f.write(f"    - Top 1 retrieved ID: `{ex['top_1_id']}` (Score: {ex['top_1_score']})\n\n")

            f.write("## 4. Kết Luận & Định Hướng Cải Thiện\n\n")
            f.write("1. **BM25:** Xử lý nhóm lỗi B nhờ cơ chế *Term Frequency Saturation* và *Document Length Normalization* ($b$).\n")
            f.write("2. **Dense Retrieval:** Xử lý nhóm lỗi A nhờ ánh xạ ngữ nghĩa sang không gian vector liên tục.\n")

        tmp_report.replace(output_report_path)
    finally:
        if tmp_report.exists():
            tmp_report.unlink(missing_ok=True)

    print(f"\nĐã xuất báo cáo chi tiết: {output_report_path}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phân tích lỗi chuyên sâu (Error Analysis) cho mô hình TF-IDF."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Đường dẫn file tfidf_test_predictions.json.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Đường dẫn file parquet corpus preprocessed.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_OUTPUT_REPORT_PATH,
        help="Đường dẫn lưu file markdown báo cáo phân tích lỗi.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Ngưỡng tỷ lệ trùng từ để phân định Lexical Chasm [0.0, 1.0] (mặc định: 0.35).",
    )
    args = parser.parse_args()

    exit_code = analyze_tfidf_errors(
        predictions_path=args.predictions,
        corpus_path=args.corpus,
        output_report_path=args.output_report,
        overlap_threshold=args.threshold,
    )
    sys.exit(exit_code)