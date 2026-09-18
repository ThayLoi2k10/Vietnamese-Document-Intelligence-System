# scripts/retrieval/build_tfidf_index.py
"""Script xây dựng và tuần tự hóa chỉ mục TF-IDF từ Corpus điều luật pháp lý.

Thực hiện:
    1. Kiểm tra idempotency (bỏ qua nếu index đã tồn tại, trừ khi có cờ --force)
    2. Đọc tối ưu các cột cần thiết từ corpus_articles_preprocessed.parquet
    3. Guard kích thước corpus (chống rỗng và đảm bảo đủ >= 5 documents cho sanity test)
    4. Huấn luyện TF-IDF Vectorizer và xác thực 100% article_id mapping khớp với corpus
    5. Đo lường kích thước từ vựng, độ thưa (sparsity), thời gian fit và dung lượng
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
from src.retrieval.tfidf_retriever import TfidfRetriever


DEFAULT_CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_retrieval"
    / "corpus_articles_preprocessed.parquet"
)
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "indices" / "tfidf_index.joblib"


def build_tfidf_index(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    output_path: Path = DEFAULT_INDEX_PATH,
    ngram_max: int = 1,
    min_df: int = 1,
    max_df: float = 0.9,
    force: bool = False,
) -> int:
    """Xây dựng và lưu trữ chỉ mục TF-IDF."""
    print("=" * 60)
    print("BƯỚC 5.4: XÂY DỰNG CHỈ MỤC TF-IDF (CORPUS INDEXING)")
    print("=" * 60)

    if not corpus_path.exists():
        print(f"[FAIL] Không tìm thấy corpus tại: {corpus_path}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Kiểm tra Idempotency
    if not force and output_path.exists() and output_path.stat().st_size > 0:
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(
            f"Chỉ mục TF-IDF đã tồn tại: {output_path} ({file_size_mb:.2f} MB).\n"
            "Bỏ qua bước tạo lại (Dùng cờ --force nếu muốn re-index)."
        )
        return 0

    # 2. Đọc dữ liệu (chỉ chiếu các cột cần thiết để tiết kiệm RAM)
    print(f"1. Đọc corpus: {corpus_path.name}...")
    start_time = time.time()
    try:
        corpus_table = pq.read_table(
            corpus_path, columns=["global_article_id", "text"]
        )
        corpus_df = corpus_table.to_pandas()
    except Exception as e:
        print(f"[FAIL] Lỗi nạp corpus: {e}")
        return 1

    load_duration = time.time() - start_time
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

    # 4. Huấn luyện TF-IDF Vectorizer
    ngram_range = (1, ngram_max)
    print(
        f"\n2. Huấn luyện TF-IDF Vectorizer (ngram_range={ngram_range}, min_df={min_df}, max_df={max_df})..."
    )
    fit_start = time.time()
    retriever = TfidfRetriever(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )
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
        print("[FAIL] Article ID mapping không khớp corpus.")
        return 1
    print("   -> [PASS] Article ID mapping khớp 100% với corpus.")

    vocab_size = len(retriever.vectorizer.vocabulary_)
    matrix_shape = retriever.tfidf_matrix.shape
    nnz = retriever.tfidf_matrix.nnz
    sparsity = (1.0 - (nnz / (matrix_shape[0] * matrix_shape[1]))) * 100

    print(f"   -> Thời gian fit ma trận: {fit_duration:.2f}s")
    print(f"   -> Kích thước từ vựng (Vocabulary): {vocab_size:,} terms")
    print(f"   -> Hình dạng ma trận (Docs x Terms): {matrix_shape}")
    print(f"   -> Số phần tử khác 0 (NNZ): {nnz:,}")
    print(f"   -> Độ thưa ma trận (Sparsity): {sparsity:.4f}%")

    # 5. Lưu chỉ mục ra disk
    print(f"\n3. Lưu chỉ mục ra disk: {output_path}...")
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

    # 6. Sanity Test truy vấn mẫu
    print("\n4. Sanity Test: Kiểm tra truy vấn mẫu...")
    test_query = "thời hiệu khởi kiện tranh chấp hợp đồng thương mại"
    print(f"   Query: '{test_query}'")
    test_start = time.time()
    top_results = retriever.search(test_query, top_k=5)
    query_duration = (time.time() - test_start) * 1000

    print(f"   -> Thời gian truy vấn: {query_duration:.2f} ms")
    print("   -> Kết quả Top 5:")
    for res in top_results:
        print(f"      Rank {res.rank}: article_id = {res.article_id:<6} | score = {res.score:.4f}")

    # Kiểm tra điều kiện đầu ra có kiểm soát
    if len(top_results) != 5:
        print(f"[FAIL] Số lượng kết quả trả về không đúng top_k=5 (thực tế: {len(top_results)}).")
        return 1

    if not all(res.score >= 0 for res in top_results):
        print("[FAIL] Điểm tương đồng Cosine âm.")
        return 1

    for i in range(len(top_results) - 1):
        if top_results[i].score < top_results[i + 1].score:
            print("[FAIL] Điểm số giữa các rank không sắp xếp giảm dần.")
            return 1

    print("\n" + "=" * 60)
    print("HOÀN THÀNH XÂY DỰNG CHỈ MỤC TF-IDF THÀNH CÔNG.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Huấn luyện và lập chỉ mục TF-IDF cho Corpus điều luật."
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
        help="Đường dẫn lưu file joblib của chỉ mục TF-IDF.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=1,
        help="Ngưỡng trên của ngram_range (mặc định 1: unigram).",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=1,
        help="Số tài liệu tối thiểu từ phải xuất hiện (mặc định: 1).",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.9,
        help="Tỷ lệ tài liệu tối đa xuất hiện từ (mặc định: 0.9).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Buộc huấn luyện và ghi đè index ngay cả khi đã tồn tại.",
    )
    args = parser.parse_args()

    exit_code = build_tfidf_index(
        corpus_path=args.corpus,
        output_path=args.output,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        max_df=args.max_df,
        force=args.force,
    )
    sys.exit(exit_code)