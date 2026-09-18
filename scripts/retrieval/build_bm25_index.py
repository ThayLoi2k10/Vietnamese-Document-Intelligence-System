# scripts/retrieval/build_bm25_index.py
"""Script xây dựng và tuần tự hóa chỉ mục BM25 (BM25Okapi) từ Corpus điều luật.

Thực hiện:
    1. Kiểm tra idempotency (bỏ qua nếu index đã tồn tại, trừ khi có cờ --force)
    2. Đọc tối ưu các cột cần thiết từ corpus_articles_preprocessed.parquet
    3. Guard kích thước corpus (chống rỗng và đảm bảo đủ >= 5 documents cho sanity test)
    4. Xây dựng Inverted Index và xác thực 100% article_id mapping khớp với corpus
    5. Đo lường kích thước từ vựng, số postings, avgdl, thời gian fit và dung lượng
    6. Lưu an toàn chỉ mục ra thư mục data/indices/
    7. Chạy sanity test truy vấn mẫu để nghiệm thu chất lượng xếp hạng
"""

import argparse
from pathlib import Path
import sys
import time

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

import pyarrow.parquet as pq
from src.retrieval.bm25_retriever import BM25Retriever


DEFAULT_CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_retrieval"
    / "corpus_articles_preprocessed.parquet"
)
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "indices" / "bm25_index.pkl"


def build_bm25_index(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    output_path: Path = DEFAULT_INDEX_PATH,
    k1: float = 1.5,
    b: float = 0.75,
    force: bool = False,
) -> int:
    """Xây dựng và lưu trữ chỉ mục BM25."""
    print("=" * 65)
    print("BƯỚC 5.7: XÂY DỰNG CHỈ MỤC BM25 (BM25OKAPI INDEXING)")
    print("=" * 65)

    if not corpus_path.exists():
        print(f"[FAIL] Không tìm thấy corpus tại: {corpus_path}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Kiểm tra Idempotency
    if not force and output_path.exists() and output_path.stat().st_size > 0:
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(
            f"Chỉ mục BM25 đã tồn tại: {output_path} ({file_size_mb:.2f} MB).\n"
            "Bỏ qua bước tạo lại (Dùng cờ --force nếu muốn re-index)."
        )
        return 0

    # 2. Đọc dữ liệu (chỉ chiếu cột cần thiết để tối ưu RAM)
    print(f"1. Đọc corpus: {corpus_path.name}...")
    start_load = time.time()
    try:
        corpus_table = pq.read_table(
            corpus_path, columns=["global_article_id", "text"]
        )
        corpus_df = corpus_table.to_pandas()
    except Exception as e:
        print(f"[FAIL] Lỗi nạp corpus: {e}")
        return 1

    load_duration = time.time() - start_load
    print(f"   -> Đã nạp {len(corpus_df):,} điều luật trong {load_duration:.2f}s")

    # 3. Guard kích thước corpus
    if len(corpus_df) == 0:
        print("[FAIL] Corpus rỗng, không thể lập chỉ mục.")
        return 1

    if len(corpus_df) < 5:
        print(
            f"[FAIL] Corpus chỉ có {len(corpus_df)} documents, "
            "không đủ để chạy sanity test top_k=5."
        )
        return 1

    # 4. Xây dựng Inverted Index
    print(f"\n2. Tiến hành tokenize và xây dựng Inverted Index (k1={k1}, b={b})...")
    fit_start = time.time()
    retriever = BM25Retriever(k1=k1, b=b)
    retriever.fit(corpus_df)
    fit_duration = time.time() - fit_start

    # Xác thực article_id mapping nội bộ của retriever
    expected_ids = corpus_df["global_article_id"].tolist()
    actual_ids = (
        retriever.article_ids.tolist()
        if hasattr(retriever.article_ids, "tolist")
        else list(retriever.article_ids)
    )
    if actual_ids != expected_ids:
        print("[FAIL] Article ID mapping của BM25Retriever không khớp corpus.")
        return 1
    print("   -> [PASS] Article ID mapping khớp 100% với corpus.")

    vocab_size = len(retriever.inverted_index)
    total_postings = sum(len(p["docs"]) for p in retriever.inverted_index.values())

    print(f"   -> Thời gian lập chỉ mục: {fit_duration:.2f}s")
    print(f"   -> Kích thước từ vựng (Vocabulary): {vocab_size:,} terms")
    print(f"   -> Tổng số bản ghi postings: {total_postings:,}")
    print(f"   -> Độ dài tài liệu trung bình (avgdl): {retriever.avgdl:.2f} words")

    # 5. Lưu trữ chỉ mục ra đĩa
    print(f"\n3. Lưu trữ chỉ mục ra đĩa: {output_path}...")
    save_start = time.time()
    try:
        retriever.save(output_path)
    except Exception as e:
        print(f"[FAIL] Lỗi khi lưu index: {e}")
        return 1

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(
        f"   -> Lưu thành công trong {time.time() - save_start:.2f}s (Dung lượng: {file_size_mb:.2f} MB)"
    )

    # 6. Sanity Test kiểm chứng thứ hạng và tốc độ
    print("\n4. Sanity Test: Kiểm tra truy vấn mẫu...")
    test_query = "thời hiệu khởi kiện tranh chấp hợp đồng thương mại"
    print(f"   Query: '{test_query}'")

    q_start = time.perf_counter()
    top_results = retriever.search(test_query, top_k=5)
    q_latency = (time.perf_counter() - q_start) * 1000

    print(f"   -> Thời gian truy vấn: {q_latency:.2f} ms")
    print("   -> Kết quả Top 5:")
    for res in top_results:
        print(f"      Rank {res.rank}: article_id = {res.article_id:<6} | score = {res.score:.4f}")

    # Kiểm tra điều kiện đầu ra có kiểm soát
    if len(top_results) != 5:
        print(f"[FAIL] Số lượng kết quả không đúng top_k=5 (thực tế: {len(top_results)}).")
        return 1

    if not all(res.score > 0 for res in top_results):
        print("[FAIL] Điểm BM25 không dương trên truy vấn có từ khóa khớp.")
        return 1

    for i in range(len(top_results) - 1):
        if top_results[i].score < top_results[i + 1].score:
            print("[FAIL] Thứ hạng không giảm dần theo điểm số BM25.")
            return 1

    print("\n" + "=" * 65)
    print("HOÀN THÀNH XÂY DỰNG CHỈ MỤC BM25 THÀNH CÔNG.")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Huấn luyện và lập chỉ mục BM25 cho Corpus điều luật."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Đường dẫn file parquet corpus preprocessed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Đường dẫn lưu file pkl của chỉ mục BM25.",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="Siêu tham số k1 điều chỉnh độ bão hòa term frequency (mặc định: 1.5).",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="Siêu tham số b điều chỉnh phạt độ dài tài liệu (mặc định: 0.75).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Buộc lập chỉ mục và ghi đè ngay cả khi index đã tồn tại.",
    )
    args = parser.parse_args()

    exit_code = build_bm25_index(
        corpus_path=args.corpus,
        output_path=args.output,
        k1=args.k1,
        b=args.b,
        force=args.force,
    )
    sys.exit(exit_code)