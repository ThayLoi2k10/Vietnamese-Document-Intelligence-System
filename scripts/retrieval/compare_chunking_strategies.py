# scripts/retrieval/compare_chunking_strategies.py
"""Script thực thi ma trận thực nghiệm phân đoạn (Phase 7 Experimental Matrix).

Tuân thủ nghiêm ngặt:
    - Mục 9: Experimental Matrix (C0..C9, Tier 1, Tier 2, Tier 3)
    - Mục 10: Fixed vs Independent Variables
    - Mục 15, 16, 17: Chunk retrieval, max-score aggregation, search depth >= 250
    - Mục 18: Retrieval, Engineering, and Diagnostic Metrics
    - Mục 19: Tuyệt đối KHÔNG sử dụng retrieval_test.json để chọn lựa mô hình
    - Đánh giá trên tập Predeclared Dev (retrieval_dev_val.json)
    - Xuất kết quả: results/benchmarks/chunking_experiment_results.json
"""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pyarrow.parquet as pq
import faiss
import torch
from transformers import AutoTokenizer

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

from src.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    evaluate_query_retrieval,
)
from src.retrieval.chunking import (
    BaseChunker,
    FixedSizeChunker,
    ParagraphChunker,
    RecursiveChunker,
    SentenceChunker,
    StructureAwareChunker,
)
from src.retrieval.dense.encoder import DenseEncoder

DEFAULT_CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "legal_retrieval"
    / "corpus_articles_preprocessed.parquet"
)
DEFAULT_DEV_VAL_PATH = PROJECT_ROOT / "data" / "benchmarks" / "retrieval_dev_val.json"
DEFAULT_ARTICLE_EMBEDDINGS_PATH = (
    PROJECT_ROOT / "data" / "indices" / "dense_bge_m3" / "embeddings.f32.npy"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmarks"
CHUNK_SEARCH_DEPTH = 300  # Đảm bảo đủ >= 100 unique articles sau max aggregation
EVAL_K_LIST = [1, 5, 10, 20, 50, 100]


def get_or_create_query_embeddings(
    queries: List[str],
    encoder: DenseEncoder,
    cache_path: Path,
) -> np.ndarray:
    """Lấy hoặc tính toán query embeddings cho tập dev."""
    if cache_path.exists():
        print(f"   [CACHE] Nạp query embeddings từ: {cache_path.name}")
        return np.load(cache_path)

    print(f"   -> Mã hóa {len(queries)} câu hỏi dev bằng BGE-M3...")
    t0 = time.time()
    vecs = encoder.encode_queries(queries, batch_size=32)
    dur = time.time() - t0
    print(f"      Hoàn thành mã hóa query trong {dur:.2f}s ({dur/len(queries)*1000:.2f} ms/q)")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, vecs)
    return vecs


def build_chunk_embeddings_matrix(
    chunks: List[Any],
    article_embeddings: np.ndarray,
    corpus_df: Any,
    encoder: DenseEncoder,
    cache_dict: Dict[str, np.ndarray],
    cache_file: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Xây dựng ma trận embeddings cho các chunk bằng cách tái sử dụng embedding bài viết và bộ đệm băm nội dung."""
    n_chunks = len(chunks)
    chunk_matrix = np.zeros((n_chunks, 1024), dtype=np.float32)
    chunk_to_art_id = np.zeros(n_chunks, dtype=np.int32)

    # Lấy độ dài nguyên thủy của các điều luật để nhận biết chunk nguyên vẹn
    art_len_map = {
        int(r["global_article_id"]): len(str(r["text"])) if r["text"] else 0
        for _, r in corpus_df.iterrows()
    }

    chunks_to_encode: List[Tuple[int, str]] = []  # (matrix_idx, chunk_text)

    for i, c in enumerate(chunks):
        art_id = c.global_article_id
        chunk_to_art_id[i] = art_id

        # Nếu điều luật chỉ có 1 chunk duy nhất bao trọn vẹn văn bản: dùng trực tiếp embedding Phase 6
        if c.total_chunks == 1 and c.char_start == 0 and c.char_end == art_len_map.get(art_id, 0):
            chunk_matrix[i] = article_embeddings[art_id]
        elif c.content_hash in cache_dict:
            chunk_matrix[i] = cache_dict[c.content_hash]
        else:
            chunks_to_encode.append((i, c.chunk_text))

    if chunks_to_encode:
        print(f"      Cần mã hóa {len(chunks_to_encode):,} chunk văn bản mới (không trùng lặp)...")
        t_enc = time.time()
        texts = [item[1] for item in chunks_to_encode]
        batch_size = 32
        max_len = 512

        # Mã hóa trực tiếp bằng encoder.model để kiểm soát max_length và batch_size tối ưu cho GPU
        new_vecs_list = []
        for b_i in range(0, len(texts), batch_size):
            b_texts = texts[b_i : b_i + batch_size]
            safe_b = [t if t else "" for t in b_texts]
            out = encoder.model.encode(
                safe_b,
                batch_size=len(safe_b),
                max_length=max_len,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            new_vecs_list.append(out["dense_vecs"].astype(np.float32))
            if (b_i // batch_size) % 100 == 0 and b_i > 0:
                print(f"         ... đã mã hóa {b_i:,}/{len(texts):,} chunks ({b_i/(time.time()-t_enc):.1f} chunks/s)")

        new_vecs = np.concatenate(new_vecs_list, axis=0)

        for (m_idx, _), v in zip(chunks_to_encode, new_vecs):
            chunk_matrix[m_idx] = v
            c_hash = chunks[m_idx].content_hash
            cache_dict[c_hash] = v

        enc_dur = time.time() - t_enc
        speed = len(chunks_to_encode) / enc_dur if enc_dur > 0 else 0
        print(f"      Đã mã hóa trong {enc_dur:.2f}s ({speed:.1f} chunks/s)")

        # Lưu cập nhật cache
        np.savez_compressed(cache_file, **cache_dict)

    # Chuẩn hóa L2 phòng ngừa sai số
    norms = np.linalg.norm(chunk_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    chunk_matrix /= norms

    return chunk_matrix, chunk_to_art_id


def evaluate_chunk_retrieval(
    index: faiss.IndexFlatIP,
    chunk_to_art_id: np.ndarray,
    query_embeddings: np.ndarray,
    ground_truth: List[Set[int]],
    search_depth: int = CHUNK_SEARCH_DEPTH,
) -> Tuple[Dict[str, float], float, List[Dict[str, float]]]:
    """Tìm kiếm vector chunk, tổng hợp max-score về bài viết và tính metrics IR."""
    t_search_start = time.perf_counter()
    scores_mat, indices_mat = index.search(query_embeddings, search_depth)
    search_duration = time.perf_counter() - t_search_start

    num_queries = len(ground_truth)
    per_query_metrics = []

    for q_idx in range(num_queries):
        row_indices = indices_mat[q_idx]
        row_scores = scores_mat[q_idx]
        rel_set = ground_truth[q_idx]

        # Tổng hợp max-score theo global_article_id
        article_score_map: Dict[int, float] = {}
        for c_idx, score in zip(row_indices, row_scores):
            if c_idx < 0:
                continue
            art_id = int(chunk_to_art_id[c_idx])
            curr_max = article_score_map.get(art_id, -1e9)
            if score > curr_max:
                article_score_map[art_id] = float(score)

        # Xếp hạng giảm dần theo score
        sorted_articles = sorted(
            article_score_map.items(), key=lambda x: x[1], reverse=True
        )
        ranked_art_ids = [item[0] for item in sorted_articles[:100]]

        q_metrics = evaluate_query_retrieval(
            retrieved_ids=ranked_art_ids,
            relevant_ids=rel_set,
            k_list=EVAL_K_LIST,
        )
        per_query_metrics.append(q_metrics)

    macro_metrics = aggregate_retrieval_metrics(per_query_metrics)
    avg_latency_ms = (search_duration / num_queries) * 1000.0

    return macro_metrics, avg_latency_ms, per_query_metrics


def run_chunking_experiments(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    dev_path: Path = DEFAULT_DEV_VAL_PATH,
    article_embeddings_path: Path = DEFAULT_ARTICLE_EMBEDDINGS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    strategies_filter: Optional[str] = None,
) -> int:
    print("=" * 75)
    print("PHASE 7: SO SÁNH VÀ ĐÁNH GIÁ CÁC CHIẾN LƯỢC PHÂN ĐOẠN (TIER 1, 2, 3)")
    print("=" * 75)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "chunking_experiment_results.json"

    # Nạp kết quả cũ nếu đã tồn tại để tiếp tục tiến trình
    experiment_results: Dict[str, Any] = {}
    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                experiment_results = json.load(f)
            print(f"   [CHECKPOINT] Đã tìm thấy kết quả cũ với {len(experiment_results.get('candidates', {}))} ứng viên đã hoàn thành.")
        except Exception:
            pass

    # 1. Nạp tài nguyên nền tảng
    print("1. Nạp corpus và embeddings tham chiếu Phase 6...")
    corpus_table = pq.read_table(corpus_path)
    corpus_df = corpus_table.to_pandas()
    num_articles = len(corpus_df)
    print(f"   -> Corpus: {num_articles:,} điều luật")

    if not article_embeddings_path.exists():
        print(f"[FAIL] Không tìm thấy Phase 6 embeddings tại: {article_embeddings_path}")
        return 1
    article_embeddings = np.load(article_embeddings_path)
    print(f"   -> Phase 6 embeddings: {article_embeddings.shape}")

    # 2. Nạp tập Dev Benchmark (351 queries)
    print(f"2. Nạp Dev Benchmark: {dev_path.name}...")
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_queries = json.load(f)
    print(f"   -> Số lượng câu hỏi đánh giá: {len(dev_queries):,}")
    query_texts = [q["query_text"] for q in dev_queries]
    ground_truth = [set(q["relevant_article_ids"]) for q in dev_queries]

    # 3. Khởi tạo Encoder & Tokenizer
    print("3. Khởi tạo BGE-M3 Encoder...")
    encoder = DenseEncoder()
    tokenizer = encoder.model.tokenizer

    # Query embeddings cache
    query_cache_path = output_dir / "dev_val_bge_m3_query_embeddings.npy"
    dev_query_vecs = get_or_create_query_embeddings(
        query_texts, encoder, query_cache_path
    )

    # Chunk embeddings cache
    chunk_cache_file = output_dir / "chunk_embeddings_cache.npz"
    chunk_cache_dict: Dict[str, np.ndarray] = {}
    if chunk_cache_file.exists():
        try:
            loaded_cache = np.load(chunk_cache_file)
            for k in loaded_cache.files:
                chunk_cache_dict[k] = loaded_cache[k]
            print(f"   [CACHE] Nạp {len(chunk_cache_dict):,} vector chunk từ cache.")
        except Exception:
            pass

    # 4. Thiết lập danh sách các ứng viên chiến lược (Tier 1 & Tier 2)
    candidate_strategies: List[Tuple[str, BaseChunker, str]] = [
        ("C2_fixed_512", FixedSizeChunker(target_size=512, overlap=0.0, tokenizer=tokenizer), "Fixed-size 512"),
        ("C5_recursive_512", RecursiveChunker(target_size=512, overlap=0.0, tokenizer=tokenizer), "Recursive 512"),
        ("C7_paragraph_512", ParagraphChunker(target_size=512, overlap=0.0, tokenizer=tokenizer), "Paragraph-based 512"),
        ("C8_structure_clause_first", StructureAwareChunker(target_size=512, clause_first_only=True, tokenizer=tokenizer), "Structure Clause-first"),
        ("C9_structure_hierarchical_512", StructureAwareChunker(target_size=512, clause_first_only=False, tokenizer=tokenizer), "Structure Hierarchical 512"),
        ("C6_sentence_512", SentenceChunker(target_size=512, overlap=0.0, tokenizer=tokenizer), "Sentence-based 512"),
        ("C3_fixed_1024", FixedSizeChunker(target_size=1024, overlap=0.0, tokenizer=tokenizer), "Fixed-size 1024"),
        ("C4_fixed_512_ov10", FixedSizeChunker(target_size=512, overlap=0.10, tokenizer=tokenizer), "Fixed-size 512 (ov 10%)"),
        ("C1_fixed_256", FixedSizeChunker(target_size=256, overlap=0.0, tokenizer=tokenizer), "Fixed-size 256"),
    ]

    if strategies_filter:
        allowed = {s.strip() for s in strategies_filter.split(",") if s.strip()}
        candidate_strategies = [c for c in candidate_strategies if c[0] in allowed]
        print(f"   [FILTER] Chỉ chạy {len(candidate_strategies)} chiến lược được chỉ định: {allowed}")

    # 5. Đánh giá điều kiện đối chuẩn C0 (Article-level reference trên tập Dev)
    print("\n" + "=" * 75)
    print("ĐÁNH GIÁ ĐIỀU KIỆN C0: ARTICLE-LEVEL REFERENCE (PHASE 6)")
    print("=" * 75)
    c0_index = faiss.IndexFlatIP(1024)
    c0_index.add(article_embeddings)
    c0_chunk_to_art = np.arange(num_articles, dtype=np.int32)
    c0_metrics, c0_lat, c0_per_q = evaluate_chunk_retrieval(
        index=c0_index,
        chunk_to_art_id=c0_chunk_to_art,
        query_embeddings=dev_query_vecs,
        ground_truth=ground_truth,
        search_depth=100,
    )
    print(f"   -> C0 nDCG@10={c0_metrics['nDCG@10']:.4f} | MRR@10={c0_metrics['MRR@10']:.4f} | "
          f"R@10={c0_metrics['R@10']:.4f} | Latency={c0_lat:.2f} ms")

    if "reference_C0" not in experiment_results:
        experiment_results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        experiment_results["dev_benchmark"] = dev_path.name
        experiment_results["num_dev_queries"] = len(dev_queries)
        experiment_results["reference_C0"] = {
            "strategy": "Article-level (Phase 6 reproduction)",
            "metrics": {k: round(v, 4) for k, v in c0_metrics.items()},
            "latency_ms": round(c0_lat, 2),
            "total_units": num_articles,
        }
    if "candidates" not in experiment_results:
        experiment_results["candidates"] = {}

    print("\n" + "=" * 75)
    print(f"{'Mã':<26} | {'nDCG@10':<8} | {'MRR@10':<8} | {'R@10':<8} | {'Delta':<8} | {'Chunks':<8} | {'Lat(ms)':<8}")
    print("-" * 75)
    print(f"{'C0_article_level':<26} | {c0_metrics['nDCG@10']:.4f}   | {c0_metrics['MRR@10']:.4f}   | "
          f"{c0_metrics['R@10']:.4f}   | {'REF':<8} | {num_articles:<8} | {c0_lat:.2f}")

    best_strategy_name = ""
    best_ndcg = -1.0
    best_candidate_record = None

    for code, chunker, desc in candidate_strategies:
        if code in experiment_results["candidates"]:
            cand = experiment_results["candidates"][code]
            ndcg_val = cand["metrics"]["nDCG@10"]
            delta_ndcg = cand["delta_vs_c0"]["delta_nDCG@10"]
            print(f"{code:<26} | {ndcg_val:.4f}   | {cand['metrics']['MRR@10']:.4f}   | "
                  f"{cand['metrics']['R@10']:.4f}   | {delta_ndcg:+.4f}   | {cand['total_chunks']:<8} | (đã cache)")
            if ndcg_val > best_ndcg:
                best_ndcg = ndcg_val
                best_strategy_name = code
            continue

        t_chunk_start = time.time()
        chunks = chunker.chunk_corpus(corpus_df)
        chunk_time = time.time() - t_chunk_start

        total_chunks = len(chunks)
        tok_lens = [c.token_count for c in chunks]
        mean_tok = float(np.mean(tok_lens)) if tok_lens else 0.0
        p95_tok = float(np.percentile(tok_lens, 95)) if tok_lens else 0.0
        p99_tok = float(np.percentile(tok_lens, 99)) if tok_lens else 0.0

        # Xây dựng ma trận embeddings
        t_mat = time.time()
        chunk_matrix, chunk_to_art_id = build_chunk_embeddings_matrix(
            chunks=chunks,
            article_embeddings=article_embeddings,
            corpus_df=corpus_df,
            encoder=encoder,
            cache_dict=chunk_cache_dict,
            cache_file=chunk_cache_file,
        )
        mat_time = time.time() - t_mat

        # Xây dựng FAISS index
        c_index = faiss.IndexFlatIP(1024)
        c_index.add(chunk_matrix)

        # Đánh giá truy xuất
        macro_met, avg_lat, per_q = evaluate_chunk_retrieval(
            index=c_index,
            chunk_to_art_id=chunk_to_art_id,
            query_embeddings=dev_query_vecs,
            ground_truth=ground_truth,
            search_depth=CHUNK_SEARCH_DEPTH,
        )

        ndcg_val = macro_met["nDCG@10"]
        delta_ndcg = ndcg_val - c0_metrics["nDCG@10"]

        # Thống kê win/loss trên từng query so với C0
        win_count = sum(1 for qm, cm in zip(per_q, c0_per_q) if qm["nDCG@10"] > cm["nDCG@10"])
        loss_count = sum(1 for qm, cm in zip(per_q, c0_per_q) if qm["nDCG@10"] < cm["nDCG@10"])
        tie_count = len(per_q) - win_count - loss_count

        delta_str = f"{delta_ndcg:+.4f}"
        print(f"{code:<26} | {ndcg_val:.4f}   | {macro_met['MRR@10']:.4f}   | "
              f"{macro_met['R@10']:.4f}   | {delta_str:<8} | {total_chunks:<8} | {avg_lat:.2f}")

        cand_record = {
            "code": code,
            "description": desc,
            "strategy": chunker.strategy_name,
            "target_size": chunker.target_size,
            "overlap": chunker.overlap,
            "total_chunks": total_chunks,
            "chunks_per_article": round(total_chunks / num_articles, 2),
            "mean_token_count": round(mean_tok, 1),
            "p95_token_count": round(p95_tok, 1),
            "p99_token_count": round(p99_tok, 1),
            "chunk_generation_time_sec": round(chunk_time, 2),
            "metrics": {k: round(v, 4) for k, v in macro_met.items()},
            "delta_vs_c0": {
                "delta_nDCG@10": round(delta_ndcg, 4),
                "delta_MRR@10": round(macro_met["MRR@10"] - c0_metrics["MRR@10"], 4),
                "delta_R@10": round(macro_met["R@10"] - c0_metrics["R@10"], 4),
                "wins": win_count,
                "losses": loss_count,
                "ties": tie_count,
            },
            "engineering": {
                "latency_ms_per_query": round(avg_lat, 2),
                "throughput_qps": round(len(dev_queries) / (avg_lat * len(dev_queries) / 1000), 1),
                "index_size_mb": round((total_chunks * 1024 * 4) / (1024 * 1024), 2),
            },
        }

        experiment_results["candidates"][code] = cand_record

        if ndcg_val > best_ndcg:
            best_ndcg = ndcg_val
            best_strategy_name = code
            best_candidate_record = cand_record

        # Lưu kết quả thực nghiệm ngay lập tức sau mỗi ứng viên
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(experiment_results, f, indent=2, ensure_ascii=False)
        print(f"      [SAVED] Đã cập nhật kết quả: {report_file.name}")

        # Dọn dẹp bộ nhớ
        del c_index, chunk_matrix, chunk_to_art_id, chunks
        gc.collect()

    # Tìm chiến lược tốt nhất trong toàn bộ candidates đã hoàn thành
    for c_code, c_rec in experiment_results["candidates"].items():
        c_ndcg = c_rec["metrics"]["nDCG@10"]
        if c_ndcg > best_ndcg:
            best_ndcg = c_ndcg
            best_strategy_name = c_code

    print("=" * 75)
    print(f"\nChiến lược tốt nhất trên tập Dev theo nDCG@10: {best_strategy_name} (nDCG@10={best_ndcg:.4f})")
    experiment_results["selected_best_strategy"] = best_strategy_name

    # Lưu kết quả thực nghiệm cuối cùng
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2, ensure_ascii=False)
    print(f"[PASS] Đã ghi kết quả thực nghiệm hoàn chỉnh: {report_file.name}")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="So sánh và đánh giá các chiến lược chunking")
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--dev-path", type=Path, default=DEFAULT_DEV_VAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strategies", type=str, default=None, help="Danh sách mã chiến lược phân cách bởi dấu phẩy")
    args = parser.parse_args()

    sys.exit(
        run_chunking_experiments(
            corpus_path=args.corpus_path,
            dev_path=args.dev_path,
            output_dir=args.output_dir,
            strategies_filter=args.strategies,
        )
    )
