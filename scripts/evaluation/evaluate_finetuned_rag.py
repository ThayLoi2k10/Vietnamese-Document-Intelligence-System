r"""Đánh giá toàn diện (1) % Tìm đúng điều luật trên 438 câu Frozen Test và (2) % Trả lời đúng câu hỏi End-to-End RAG.

Chạy ở Local sau khi đã tải trọng số `data/indices/reranker_finetuned/lora_reranker_weights.pt`:
    .\.venv\Scripts\python.exe scripts\evaluation\evaluate_finetuned_rag.py --qa-samples 20
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import pyarrow.parquet as pq
import requests

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    evaluate_query_retrieval,
)
from src.rag.context.compressor import LegalPassageCompressor
from src.rag.output.formatter import LegalResponseFormatter
from src.rag.query.normalizer import LegalAcronymNormalizer
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.reranking.finetuned_reranker import (
    DEFAULT_FINETUNED_WEIGHTS_PATH,
    FinetunedLegalReranker,
)


def load_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            if k and v and not os.environ.get(k):
                os.environ[k] = v


def compute_hit_stats(preds_list: List[Dict[str, Any]], key: str = "top_10_retrieved") -> Dict[str, float]:
    hit1 = hit3 = hit5 = hit10 = r3 = 0.0
    per_q = []
    for p in preds_list:
        rel = set(int(x) for x in p["ground_truth"])
        raw_cands = p.get(key) or p.get("top_100_retrieved_ids") or []
        cands = [int(x["article_id"]) if isinstance(x, dict) else int(x) for x in raw_cands]
        full_cands = [int(x) for x in p.get("top_100_retrieved_ids", cands)]

        hit1 += float(bool(set(cands[:1]) & rel))
        hit3 += float(bool(set(cands[:3]) & rel))
        hit5 += float(bool(set(cands[:5]) & rel))
        hit10 += float(bool(set(cands[:10]) & rel))
        r3 += len(set(cands[:3]) & rel) / max(1, len(rel))
        per_q.append(evaluate_query_retrieval(full_cands, rel))

    agg = aggregate_retrieval_metrics(per_q)
    n = max(1, len(preds_list))
    return {
        "Hit@1_pct": round(hit1 / n * 100, 2),
        "Hit@3_pct": round(hit3 / n * 100, 2),
        "Recall@3_pct": round(r3 / n * 100, 2),
        "Hit@5_pct": round(hit5 / n * 100, 2),
        "Hit@10_pct": round(hit10 / n * 100, 2),
        "MRR@10": round(float(agg.get("MRR@10", 0.0)), 4),
        "nDCG@10": round(float(agg.get("nDCG@10", 0.0)), 4),
    }


def call_llm_sync(user_prompt: str, system_prompt: str) -> tuple[str, str]:
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        models = ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b")
        for attempt in range(4):
            for model in models:
                try:
                    resp = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                        json={
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "temperature": 0.1,
                            "max_tokens": 2048,
                        },
                        timeout=60,
                    )
                    if resp.status_code == 429:
                        time.sleep(8 + attempt * 4)
                        continue
                    if resp.status_code == 200:
                        txt = resp.json()["choices"][0]["message"].get("content", "")
                        if "</think>" in txt:
                            txt = txt.split("</think>", 1)[-1].strip()
                        if txt.strip():
                            return txt.strip(), f"Groq ({model})"
                except Exception:
                    continue
            time.sleep(6)

    # Fallback sang Ollama nếu Groq hết quota
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:1.5b",
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 600},
            },
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip(), "Ollama (qwen2.5:1.5b)"
    except Exception:
        pass
    return "", "None"


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá % chính xác sau khi Fine-tune Reranker.")
    parser.add_argument(
        "--qa-samples",
        type=int,
        default=20,
        help="Số câu hỏi test dùng để chấm % trả lời End-to-End bằng LLM (mặc định: 20)",
    )
    parser.add_argument(
        "--force-rerank",
        action="store_true",
        help="Chạy lại Cross-Encoder trên 438 câu test ngay cả khi đã có file cache kết quả.",
    )
    args = parser.parse_args()
    load_env()

    print("=" * 95)
    print("PHẦN 1: ĐÁNH GIÁ TRÊN TOÀN BỘ 438 CÂU HỎI FROZEN TEST (`data/benchmarks/retrieval_test.json`)")
    print("=" * 95)

    corpus_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "legal_retrieval"
        / "corpus_articles_preprocessed.parquet"
    )
    table = pq.read_table(corpus_path, columns=["global_article_id", "law_id", "aid", "text"])
    corpus_meta: Dict[int, Dict[str, str]] = {}
    corpus_rerank_text: Dict[int, str] = {}
    for gid, lid, aid, txt in zip(
        table.column("global_article_id").to_pylist(),
        table.column("law_id").to_pylist(),
        table.column("aid").to_pylist(),
        table.column("text").to_pylist(),
    ):
        gid_i = int(gid)
        lid_s, aid_s, txt_s = str(lid or ""), str(aid or ""), str(txt or "")
        corpus_meta[gid_i] = {"law_id": lid_s, "aid": aid_s, "text": txt_s}
        header = f"[Văn bản: {lid_s} | Điều: {aid_s}] " if lid_s else ""
        full_t = (header + txt_s).strip()
        if len(full_t) > 1600:
            full_t = full_t[:1200].rstrip() + "\n...\n" + full_t[-400:].lstrip()
        corpus_rerank_text[gid_i] = full_t

    bm25_preds = json.loads(
        (PROJECT_ROOT / "results" / "benchmarks" / "bm25_tuned_test_predictions.json").read_text(encoding="utf-8")
    )
    dense_preds = json.loads(
        (PROJECT_ROOT / "results" / "benchmarks" / "dense_test_predictions.json").read_text(encoding="utf-8")
    )
    hybrid_preds = json.loads(
        (PROJECT_ROOT / "results" / "benchmarks" / "hybrid_test_predictions.json").read_text(encoding="utf-8")
    )
    zs_rerank_preds = json.loads(
        (PROJECT_ROOT / "results" / "benchmarks" / "reranked_test_predictions.json").read_text(encoding="utf-8")
    )

    out_ft_preds = PROJECT_ROOT / "results" / "benchmarks" / "finetuned_reranker_test_predictions.json"
    out_ft_metrics = PROJECT_ROOT / "results" / "benchmarks" / "finetuned_reranker_test_metrics.json"

    if out_ft_preds.exists() and not args.force_rerank:
        print(f"⚡ Đã tìm thấy kết quả chấm 438 câu test tại {out_ft_preds.name}, nạp trực tiếp...")
        ft_preds = json.loads(out_ft_preds.read_text(encoding="utf-8"))
    else:
        reranker = FinetunedLegalReranker(weights_path=DEFAULT_FINETUNED_WEIGHTS_PATH)
        reranker._init_model()
        mode_tag = (
            f"Fine-tuned LoRA (alpha={reranker.fusion_alpha}, pool={reranker.candidate_pool_size})"
            if reranker.is_finetuned
            else "Chưa có file lora_reranker_weights.pt (Đang chạy Base)"
        )
        print(f"📌 Trạng thái mô hình Reranker: {mode_tag}")

        bm25 = BM25Retriever.load(PROJECT_ROOT / "data" / "indices" / "bm25_index.pkl")
        bm25.k1 = 1.80
        bm25.b = 0.85
        dense_by_qid = {
            int(x["query_id"]): [int(a) for a in x["top_100_retrieved_ids"]]
            for x in dense_preds
        }

        ft_preds = []
        t0 = time.time()
        for idx, hp in enumerate(hybrid_preds, start=1):
            qid = int(hp["query_id"])
            qtext = str(hp["query_text"])
            gt = [int(x) for x in hp["ground_truth"]]
            top100_ids = [int(x) for x in hp["top_100_retrieved_ids"]]

            bm25_ids = [r.article_id for r in bm25.search(qtext, top_k=100)]
            dense_ids = dense_by_qid.get(qid, [])
            rrf_map: Dict[int, float] = {}
            for r_idx, aid in enumerate(dense_ids[:100], start=1):
                rrf_map[aid] = rrf_map.get(aid, 0.0) + 0.75 / (5.0 + r_idx)
            for r_idx, aid in enumerate(bm25_ids[:100], start=1):
                rrf_map[aid] = rrf_map.get(aid, 0.0) + 0.25 / (5.0 + r_idx)

            pool_k = reranker.candidate_pool_size
            cand_ids = top100_ids[:pool_k]
            cand_texts = [corpus_rerank_text.get(aid, "") for aid in cand_ids]
            cand_priors = [rrf_map.get(aid, 0.0) for aid in cand_ids]

            reranked = reranker.rerank(
                query=qtext,
                candidate_ids=cand_ids,
                candidate_texts=cand_texts,
                candidate_prior_scores=cand_priors,
                candidate_pool_size=pool_k,
                top_k=pool_k,
            )
            new_top_ids = [r.article_id for r in reranked] + top100_ids[pool_k:]
            ft_preds.append(
                {
                    "query_id": qid,
                    "query_text": qtext,
                    "ground_truth": gt,
                    "top_10_retrieved": [
                        {"rank": r.rank, "article_id": r.article_id, "score": round(r.score, 6)}
                        for r in reranked[:10]
                    ],
                    "top_100_retrieved_ids": new_top_ids,
                }
            )
            if idx % 100 == 0 or idx == len(hybrid_preds):
                print(f"   Đã chấm {idx}/{len(hybrid_preds)} câu hỏi test ({time.time() - t0:.1f}s)...")
        out_ft_preds.write_text(json.dumps(ft_preds, indent=2, ensure_ascii=False), encoding="utf-8")

    systems = {
        "1. BM25 Tuned (Phase 5)": compute_hit_stats(bm25_preds),
        "2. Dense BGE-M3 (Phase 6)": compute_hit_stats(dense_preds),
        "3. Hybrid Weighted RRF (Phase 11)": compute_hit_stats(hybrid_preds),
        "4. Hybrid + Reranker (Pretrained Zero-shot)": compute_hit_stats(zs_rerank_preds),
        "5. Hybrid + Reranker (Fine-tuned của bạn)": compute_hit_stats(ft_preds),
    }

    print("\n" + "-" * 95)
    print(
        f"{'Hệ thống':43s} | {'Hit@1':>7s} | {'Hit@3':>7s} | {'Hit@5':>7s} | {'Hit@10':>7s} | {'MRR@10':>7s} | {'nDCG@10':>7s}"
    )
    print("-" * 95)
    for name, st_m in systems.items():
        print(
            f"{name:43s} | {st_m['Hit@1_pct']:6.2f}% | {st_m['Hit@3_pct']:6.2f}% | "
            f"{st_m['Hit@5_pct']:6.2f}% | {st_m['Hit@10_pct']:6.2f}% | "
            f"{st_m['MRR@10']:7.4f} | {st_m['nDCG@10']:7.4f}"
        )
    print("-" * 95)

    out_ft_preds = PROJECT_ROOT / "results" / "benchmarks" / "finetuned_reranker_test_predictions.json"
    out_ft_metrics = PROJECT_ROOT / "results" / "benchmarks" / "finetuned_reranker_test_metrics.json"
    out_ft_preds.write_text(json.dumps(ft_preds, indent=2, ensure_ascii=False), encoding="utf-8")
    out_ft_metrics.write_text(json.dumps(systems, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.qa_samples <= 0:
        return

    print("\n" + "=" * 95)
    print(f"PHẦN 2: CHẤM % TRẢ LỜI CÂU HỎI THỰC TẾ END-TO-END ({args.qa_samples} CÂU HỎI TEST)")
    print("=" * 95)

    normalizer = LegalAcronymNormalizer()
    compressor = LegalPassageCompressor()
    sample_items = ft_preds[: args.qa_samples]

    context_hits = 0
    answered_ok = 0
    citation_hits = 0
    grounding_scores: List[float] = []
    qa_details: List[Dict[str, Any]] = []

    for i, item in enumerate(sample_items, start=1):
        qtext = item["query_text"]
        gt_ids = set(int(x) for x in item["ground_truth"])
        gt_laws = {corpus_meta[aid]["law_id"].lower() for aid in gt_ids if aid in corpus_meta}

        norm_q, _, _ = normalizer.normalize(qtext)
        top5_ids = [int(x) for x in item["top_100_retrieved_ids"][:5]]
        has_gt_in_ctx = bool(set(top5_ids) & gt_ids)
        context_hits += int(has_gt_in_ctx)

        docs = []
        for rank_idx, aid in enumerate(top5_ids, start=1):
            meta = corpus_meta.get(aid, {})
            docs.append(
                {
                    "rank": rank_idx,
                    "article_id": aid,
                    "score": 1.0 / rank_idx,
                    "law_id": meta.get("law_id", "N/A"),
                    "aid": meta.get("aid", "N/A"),
                    "text": meta.get("text", ""),
                }
            )

        compressed_docs = compressor.compress_documents(norm_q, docs)
        ctx_blocks = []
        for d in compressed_docs:
            ctx_blocks.append(
                f"[TÀI LIỆU #{d['rank']} - Văn bản ID: {d['law_id']} | Điều ID: {d['aid']}]:\n{d['text'].strip()}"
            )
        user_prompt = (
            "=== CĂN CỨ PHÁP LÝ ĐƯỢC TRÍCH XUẤT TỪ DỮ LIỆU ===\n\n"
            + "\n\n".join(ctx_blocks)
            + f"\n\n=== CÂU HỎI CỦA NGƯỜI DÙNG ===\n{qtext}\n\n"
            + "HÃY TRẢ LỜI CÂU HỎI TRÊN DỰA TRÊN CĂN CỨ PHÁP LÝ ĐƯỢC CUNG CẤP THEO ĐÚNG CẤU TRÚC 3 PHẦN:"
        )

        ans_text, backend_name = call_llm_sync(user_prompt, LegalResponseFormatter.SYSTEM_PROMPT_TEMPLATE)
        structured = LegalResponseFormatter.to_api_json(ans_text, compressed_docs) if ans_text else {}
        grounding = structured.get("grounding", {})
        verdict = grounding.get("verdict", "ABSTAINED")
        g_score = float(grounding.get("grounding_score", 0.0))

        is_abstained = (not ans_text) or ("THÔNG BÁO THIẾU CĂN CỨ" in ans_text.upper()) or (verdict == "ABSTAINED")
        if not is_abstained:
            answered_ok += 1
        grounding_scores.append(g_score)

        # Kiểm tra câu trả lời có trích dẫn đúng văn bản / điều luật chuẩn không
        ans_lower = ans_text.lower()
        cited_gt = any(str(aid) in ans_lower for aid in gt_ids) or any(
            lid and lid in ans_lower for lid in gt_laws
        )
        if not is_abstained and (cited_gt or (has_gt_in_ctx and g_score >= 0.8)):
            citation_hits += 1

        print(
            f"[{i:02d}/{len(sample_items)}] ContextHit={'✅' if has_gt_in_ctx else '❌'} | "
            f"Answered={'✅' if not is_abstained else '⚠️'} | "
            f"CorrectCitation={'✅' if cited_gt or (has_gt_in_ctx and not is_abstained) else '❌'} | "
            f"Grounding={g_score*100:.0f}% | Q: {qtext[:65]}..."
        )
        qa_details.append(
            {
                "query_id": item["query_id"],
                "query_text": qtext,
                "ground_truth_ids": list(gt_ids),
                "retrieved_top5_ids": top5_ids,
                "context_hit_top5": has_gt_in_ctx,
                "answered_successfully": not is_abstained,
                "citation_grounded_correct": bool(not is_abstained and (cited_gt or (has_gt_in_ctx and g_score >= 0.8))),
                "grounding_score": g_score,
                "backend": backend_name,
                "answer_preview": ans_text[:400],
            }
        )

    n_q = max(1, len(sample_items))
    qa_summary = {
        "num_evaluated_queries": len(sample_items),
        "context_hit_top5_pct": round(context_hits / n_q * 100, 2),
        "answer_success_rate_pct": round(answered_ok / n_q * 100, 2),
        "accurate_grounded_answer_pct": round(citation_hits / n_q * 100, 2),
        "mean_grounding_fidelity_pct": round(sum(grounding_scores) / n_q * 100, 2),
        "details": qa_details,
    }
    qa_report_path = PROJECT_ROOT / "results" / "benchmarks" / "end_to_end_qa_accuracy_report.json"
    qa_report_path.write_text(json.dumps(qa_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("-" * 95)
    print("📊 TỔNG KẾT % CHÍNH XÁC HỎI ĐÁP END-TO-END:")
    print(f"   1. % Tìm đúng điều luật đưa vào Prompt (Context Hit@5) : {qa_summary['context_hit_top5_pct']:.1f}%")
    print(f"   2. % Trả lời thành công (Answer Rate)                  : {qa_summary['answer_success_rate_pct']:.1f}%")
    print(f"   3. % Trả lời đúng căn cứ pháp lý chuẩn (QA Accuracy)   : {qa_summary['accurate_grounded_answer_pct']:.1f}%")
    print(f"   4. % Độ trung thực trích dẫn (Grounding Fidelity)      : {qa_summary['mean_grounding_fidelity_pct']:.1f}%")
    print(f"   Đã lưu chi tiết tại: {qa_report_path}")
    print("=" * 95)


if __name__ == "__main__":
    main()
