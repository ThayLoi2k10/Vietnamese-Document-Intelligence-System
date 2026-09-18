# scripts/retrieval/run_dense_retrieval.py
"""CLI chạy truy vấn tìm kiếm tương tác với mô hình Dense Retrieval."""

import argparse
from pathlib import Path
import sys
import time
from typing import List

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

import pyarrow.parquet as pq
from src.retrieval.dense.encoder import DenseEncoder
from src.retrieval.dense.retriever import DenseRetriever

DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "indices" / "dense_bge_m3"
DEFAULT_CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_retrieval"
    / "corpus_articles_preprocessed.parquet"
)


def run_retrieval(
    query: str,
    index_dir: Path = DEFAULT_INDEX_DIR,
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    top_k: int = 10,
) -> None:
    print(f"Loading retriever from: {index_dir}...")
    t0 = time.time()
    encoder = DenseEncoder()
    retriever = DenseRetriever.load(index_dir, encoder=encoder)
    print(f"Loaded retriever in {time.time() - t0:.2f}s.")

    print(f"\nQuery: {query}")
    t_start = time.time()
    results = retriever.search(query, top_k=top_k)
    search_time = (time.time() - t_start) * 1000.0

    print(f"Search completed in {search_time:.2f} ms. Top {len(results)} results:\n")

    # Load corpus for snippets
    corpus_dict = {}
    if corpus_path.exists():
        table = pq.read_table(corpus_path, columns=["global_article_id", "text"])
        df = table.to_pandas()
        corpus_dict = dict(zip(df["global_article_id"], df["text"]))

    for res in results:
        text = corpus_dict.get(res.article_id, "")
        snippet = text.replace("\n", " ")[:120] + "..." if len(text) > 120 else text
        print(f"  [Rank {res.rank:2d}] Article ID: {res.article_id:5d} | Score: {res.score:.4f}")
        if snippet:
            print(f"             Snippet: {snippet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy truy vấn Dense Retrieval")
    parser.add_argument("--query", type=str, required=True, help="Câu hỏi truy vấn")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--top-k", type=int, default=10)

    args = parser.parse_args()
    run_retrieval(
        query=args.query,
        index_dir=args.index_dir,
        corpus_path=args.corpus_path,
        top_k=args.top_k,
    )
