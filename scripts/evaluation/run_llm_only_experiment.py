"""Execute LLM-only benchmark experiment runner without retrieval context."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

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

from src.rag.abstention.detection.abstention_detector import AbstentionDetector
from src.rag.abstention.policies.abstention_validator import AbstentionValidator
from src.rag.citation.validation.citation_validator import CitationValidator
from src.rag.config.loaders.config_loader import ConfigLoader
from src.rag.config.schemas.config_models import ExperimentConfig, SplitRole
from src.rag.errors.exceptions import (
    LLMGenerationError,
    MalformedStructuredOutputError,
)
from src.rag.evaluation.backends.rule_based_evaluator import (
    RuleBasedAbstentionCorrectnessEvaluator,
)
from src.rag.evaluation.interfaces.evaluator import RAGEvaluator
from src.rag.evaluation.metrics.models import EvaluationInput, EvaluationReport
from src.rag.generation.adapters.generic_llm_adapter import GenericLLMAdapter
from src.rag.generation.domain.models import GenerationParameters
from src.rag.generation.mocks.mock_generator import MockLLMGenerator
from src.rag.output.parsers.structured_parser import OutputParser
from src.rag.pipeline.llm_only_pipeline import LLMOnlyPipeline
from src.rag.pipeline.models import RAGResponse
from src.rag.prompting.builders.prompt_builder import PromptBuilder
from src.rag.reproducibility.manifest import ExperimentManifest
from scripts.evaluation.run_rag_experiment import (
    build_real_llm_inference_fn,
    reconstruct_rag_response_from_dict,
)


def atomic_write_json(file_path: Path, data: Any) -> None:
    """Ghi dữ liệu JSON an toàn thông qua file tạm và atomic replace."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=file_path.parent, delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp_name = tmp.name
    os.replace(tmp_name, file_path)


def load_checkpoint(
    checkpoint_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Nạp checkpoint an toàn, hỗ trợ dict và list."""
    if not checkpoint_path.exists():
        return [], []
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data, []
        elif isinstance(data, dict):
            predictions = data.get("predictions", [])
            failures = data.get("failures", [])
            return predictions, failures
        else:
            return [], []
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể đọc checkpoint từ {checkpoint_path}: {e}")
        return [], []


def save_checkpoint(
    checkpoint_path: Path,
    experiment_id: str,
    predictions: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> None:
    """Lưu checkpoint nguyên tử để tránh hỏng dữ liệu khi ngắt tiến trình."""
    checkpoint_data = {
        "experiment_id": experiment_id,
        "last_updated": time.time(),
        "completed_count": len(predictions),
        "failed_count": len(failures),
        "total_processed": len(predictions) + len(failures),
        "predictions": predictions,
        "failures": failures,
    }
    atomic_write_json(checkpoint_path, checkpoint_data)


def load_phase10_dataset(
    config: ExperimentConfig, sample_limit_override: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Nạp tập dữ liệu đối chuẩn Phase 10."""
    split_path = PROJECT_ROOT / config.dataset.split_path
    if not split_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy tệp dữ liệu benchmark Phase 10: {split_path}"
        )

    with open(split_path, "r", encoding="utf-8") as f:
        all_queries = json.load(f)

    # Nếu dùng file retrieval_train gốc thay vì phase10_comparison_dataset
    if "retrieval_train.json" in str(split_path):
        if config.dataset.split_role == SplitRole.DEVELOPMENT:
            selected_queries = all_queries[:1401]
        elif config.dataset.split_role == SplitRole.VALIDATION:
            selected_queries = all_queries[1401:1752]
        else:
            selected_queries = all_queries
    else:
        selected_queries = all_queries

    limit = (
        sample_limit_override
        if sample_limit_override is not None
        else config.dataset.sample_limit
    )
    if limit is not None and limit > 0:
        selected_queries = selected_queries[:limit]

    return selected_queries


def run_llm_only_experiment(
    config_path: Path,
    device: str = "auto",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    sample_limit_override: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_freq: int = 10,
    resume: bool = True,
    retry_failed: bool = True,
    dry_run: bool = False,
    output_dir: Path = PROJECT_ROOT / "results" / "benchmarks",
    artifacts_dir: Path = PROJECT_ROOT / "artifacts",
) -> None:
    """Thực thi toàn bộ quy trình experiment runner cho System A (LLM-only)."""
    print("=" * 80)
    print(f"BẮT ĐẦU CHẠY THỰC NGHIỆM LLM-ONLY PHASE 10 (SYSTEM A): {config_path}")
    print("=" * 80)

    # 1. Nạp và kiểm định cấu hình
    config: ExperimentConfig = ConfigLoader.from_file(
        config_path, validate=True
    )
    print(f"[1/6] Nạp cấu hình thành công. Experiment ID: {config.experiment_id}")
    print(f"      - Dataset Split Role: {config.dataset.split_role.value}")
    print(f"      - LLM Model:          {config.generation.llm_model}")
    print(f"      - Prompt Version:     {config.prompting.prompt_version}")
    print(f"      - Retrieval Context:  HOÀN TOÀN BỊ VÔ HIỆU HÓA (ZERO CONTEXT)")
    if dry_run:
        print("      - Chế độ:             MOCK / DRY-RUN (Kiểm thử không dùng GPU)")

    # 2. Khởi tạo PromptBuilder
    print("\n[2/6] Khởi tạo PromptBuilder cho LLM-only...")
    prompt_builder = PromptBuilder(
        default_version=config.prompting.prompt_version
    )

    # 3. Khởi tạo Generator (Mock hoặc Real)
    print(f"[3/6] Khởi tạo Generator cho {config.generation.llm_model}...")
    if dry_run:
        generator = MockLLMGenerator(
            mode=MockLLMGenerator.MODE_VALID_ANSWER,
            custom_json_data={
                "answer": "Đây là câu trả lời mô phỏng của LLM-only cho mục đích kiểm thử.",
                "citations": [],
                "confidence_level": "FULL",
                "abstention_reason": None,
            },
        )
    else:
        inference_fn = build_real_llm_inference_fn(
            model_name=config.generation.llm_model,
            device=device,
            api_base=api_base,
            api_key=api_key,
        )
        generator = GenericLLMAdapter(
            inference_fn=inference_fn,
            component_version=config.generation.component_version,
            supported_models=[config.generation.llm_model],
        )

    # 4. Khởi tạo Parser, Validators và ghép LLMOnlyPipeline
    print("[4/6] Khởi tạo OutputParser, AbstentionValidator và ghép LLMOnlyPipeline...")
    output_parser = OutputParser()
    citation_validator = CitationValidator()
    abstention_detector = AbstentionDetector()
    abstention_validator = AbstentionValidator()

    pipeline = LLMOnlyPipeline(
        prompt_builder=prompt_builder,
        generator=generator,
        output_parser=output_parser,
        abstention_detector=abstention_detector,
        abstention_validator=abstention_validator,
        citation_validator=citation_validator,
        default_model_identifier=config.generation.llm_model,
        pipeline_version="llm-only-pipeline-v1.0",
    )

    # 5. Nạp dataset và thực thi có checkpoint/resume
    print("\n[5/6] Nạp dataset và thực thi truy vấn...")
    queries = load_phase10_dataset(
        config, sample_limit_override=sample_limit_override
    )
    print(
        f"      Số lượng câu hỏi: {len(queries)} câu ({config.dataset.split_role.value})"
    )

    default_checkpoint_path = (
        checkpoint_path
        if checkpoint_path is not None
        else output_dir / "checkpoints" / f"{config.experiment_id}_checkpoint.json"
    )

    predictions_record: List[Dict[str, Any]] = []
    failures_record: List[Dict[str, Any]] = []
    evaluation_inputs: List[EvaluationInput] = []

    if resume and default_checkpoint_path.exists():
        print(f"      [RESUME] Phát hiện checkpoint tại: {default_checkpoint_path}")
        predictions_record, failures_record = load_checkpoint(default_checkpoint_path)
        print(
            f"      [RESUME] Đã tải {len(predictions_record)} hoàn thành (COMPLETED), "
            f"{len(failures_record)} lỗi (FAILED)."
        )

    completed_ids = {
        p["query_id"] for p in predictions_record if "query_id" in p
    }
    failed_ids = {
        f["query_id"] for f in failures_record if "query_id" in f
    }

    # Tái tạo evaluation_inputs cho các câu hỏi đã hoàn thành trong checkpoint
    query_lookup = {
        q_item.get("query_id", i + 1): q_item
        for i, q_item in enumerate(queries)
    }
    for p_item in predictions_record:
        p_qid = p_item.get("query_id")
        if p_qid in query_lookup and "response" in p_item:
            try:
                resp_obj = reconstruct_rag_response_from_dict(p_item["response"])
                gt_ids = query_lookup[p_qid].get("relevant_article_ids", [])
                eval_in = EvaluationInput(
                    query=p_item.get("query", query_lookup[p_qid]["query_text"]),
                    rag_response=resp_obj,
                    ground_truth_article_ids=gt_ids,
                    metadata={"query_id": p_qid},
                )
                evaluation_inputs.append(eval_in)
            except Exception as rec_err:
                print(
                    f"      [CẢNH BÁO] Không thể tái tạo EvaluationInput cho Query ID {p_qid}: {rec_err}"
                )

    start_exec_time = time.time()
    processed_in_this_run = 0

    for idx, q_item in enumerate(queries, start=1):
        q_text = q_item["query_text"]
        q_id = q_item.get("query_id", idx)
        gt_ids = q_item.get("relevant_article_ids", [])

        if q_id in completed_ids:
            print(f"  -> [{idx}/{len(queries)}] Query ID {q_id}: [BỎ QUA - ĐÃ HOÀN THÀNH]")
            continue
        if q_id in failed_ids:
            if not retry_failed:
                print(f"  -> [{idx}/{len(queries)}] Query ID {q_id}: [BỎ QUA - ĐÃ GHI NHẬN LỖI]")
                continue
            print(f"  -> [{idx}/{len(queries)}] Query ID {q_id}: [THỬ LẠI CÂU HỎI LỖI TỪ CHECKPOINT] {q_text[:60]}...")
        else:
            print(f"  -> [{idx}/{len(queries)}] Query ID {q_id}: {q_text[:70]}...")

        try:
            response: RAGResponse = pipeline.run(
                query=q_text,
                prompt_version=config.prompting.prompt_version,
                model_identifier=config.generation.llm_model,
                max_output_tokens=config.generation.max_output_tokens,
                generation_parameters=GenerationParameters(
                    temperature=config.generation.temperature,
                    top_p=config.generation.top_p,
                    seed=config.generation.seed,
                    extra_params=config.generation.extra_parameters,
                ),
            )

            eval_in = EvaluationInput(
                query=q_text,
                rag_response=response,
                ground_truth_article_ids=gt_ids,
                metadata={"query_id": q_id},
            )
            evaluation_inputs.append(eval_in)

            pred_item = {
                "query_id": q_id,
                "query": q_text,
                "status": "COMPLETED",
                "response": response.to_full_audit_dict(),
            }
            # Nếu câu hỏi này từng bị lỗi trước đó, xóa khỏi failures_record
            failures_record = [f for f in failures_record if f.get("query_id") != q_id]
            failed_ids.discard(q_id)

            predictions_record.append(pred_item)
            completed_ids.add(q_id)
            processed_in_this_run += 1

        except (MalformedStructuredOutputError, LLMGenerationError) as e:
            raw_output = (
                getattr(e, "details", {}).get("raw_text")
                or getattr(e, "details", {}).get("raw_response")
            )
            print(
                f"\n  [CẢNH BÁO / LỖI ĐẦU RA] Query ID {q_id} gặp lỗi {type(e).__name__}: {e}"
            )
            failure_item = {
                "query_id": q_id,
                "query": q_text,
                "status": "FAILED",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "details": getattr(e, "details", {}),
                "raw_output": raw_output,
                "timestamp": time.time(),
            }
            # Cập nhật hoặc thêm vào danh sách lỗi
            failures_record = [f for f in failures_record if f.get("query_id") != q_id]
            failures_record.append(failure_item)
            failed_ids.add(q_id)
            processed_in_this_run += 1

            save_checkpoint(
                checkpoint_path=default_checkpoint_path,
                experiment_id=config.experiment_id,
                predictions=predictions_record,
                failures=failures_record,
            )

        if (
            checkpoint_freq > 0
            and processed_in_this_run > 0
            and (processed_in_this_run % checkpoint_freq == 0)
        ):
            save_checkpoint(
                checkpoint_path=default_checkpoint_path,
                experiment_id=config.experiment_id,
                predictions=predictions_record,
                failures=failures_record,
            )
            print(
                f"      [CHECKPOINT] Đã lưu: {len(predictions_record)} hoàn thành, "
                f"{len(failures_record)} lỗi -> {default_checkpoint_path}"
            )

    save_checkpoint(
        checkpoint_path=default_checkpoint_path,
        experiment_id=config.experiment_id,
        predictions=predictions_record,
        failures=failures_record,
    )

    total_latency = time.time() - start_exec_time
    print(
        f"\n      Hoàn thành xử lý {len(queries)} câu hỏi trong {total_latency:.2f}s "
        f"({len(predictions_record)} thành công, {len(failures_record)} thất bại)."
    )

    # 6. Đánh giá và lưu Artifacts
    print("\n[6/6] Đánh giá metrics và lưu trữ kết quả...")
    evaluators_map = {
        "abstention_correctness": RuleBasedAbstentionCorrectnessEvaluator(),
    }
    active_evaluators = [
        evaluators_map[name]
        for name in config.evaluation.evaluator_names
        if name in evaluators_map
    ]

    evaluator = RAGEvaluator(
        evaluators=active_evaluators,
        evaluator_version=config.evaluation.component_version,
    )
    report: EvaluationReport = evaluator.evaluate(
        dataset=evaluation_inputs,
        experiment_id=config.experiment_id,
        dataset_split=config.dataset.split_role.value,
        component_versions={
            "retriever": "DISABLED",
            "context": "ZERO_CONTEXT",
            "prompt": config.prompting.instruction_policy_version,
            "generator": config.generation.component_version,
        },
    )

    report.error_summary = {
        "total_benchmark_queries": len(queries),
        "completed_count": len(predictions_record),
        "failed_count": len(failures_record),
        "failure_rate": len(failures_record) / max(len(queries), 1),
        "failures": failures_record,
    }
    report.metric_configurations["total_benchmark_queries"] = len(queries)
    report.metric_configurations["evaluated_queries"] = len(evaluation_inputs)
    report.metric_configurations["failed_queries"] = len(failures_record)

    metrics_path = output_dir / f"{config.experiment_id}_metrics.json"
    predictions_path = output_dir / f"{config.experiment_id}_predictions.json"
    failures_path = output_dir / f"{config.experiment_id}_failures.json"
    report_path = (
        artifacts_dir / "evaluation-reports" / f"{config.experiment_id}_report.json"
    )
    manifest_path = (
        artifacts_dir
        / "experiment-manifests"
        / f"{config.experiment_id}_manifest.json"
    )

    atomic_write_json(metrics_path, report.metric_results)
    atomic_write_json(predictions_path, predictions_record)
    atomic_write_json(failures_path, failures_record)
    atomic_write_json(report_path, report.to_dict())

    manifest = ExperimentManifest.from_config(config)
    manifest.save(manifest_path)

    print("\n" + "=" * 80)
    print("BÁO CÁO KẾT QUẢ THỰC NGHIỆM LLM-ONLY:")
    print("=" * 80)
    for m_name, m_val in report.metric_results.items():
        print(f"  * {m_name:<28}: {m_val:.4f}")
    print(f"  * {'execution_success_rate':<28}: {len(predictions_record) / max(len(queries), 1):.4f}")
    print(f"  * {'execution_failure_rate':<28}: {len(failures_record) / max(len(queries), 1):.4f}")
    print("-" * 80)
    print(f"Metrics:       {metrics_path}")
    print(f"Predictions:   {predictions_path}")
    print(f"Failures:      {failures_path}")
    print(f"Report:        {report_path}")
    print(f"Manifest:      {manifest_path}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runner thực thi thực nghiệm LLM-only Phase 10"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/phase10_llm_only_experiment.json",
        help="Đường dẫn file cấu hình thực nghiệm JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Thiết bị tính toán (auto/cuda/cpu)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Base URL nếu sử dụng vLLM / OpenAI API",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key nếu sử dụng server",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Ghi đè số lượng mẫu câu hỏi cần chạy",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Đường dẫn file checkpoint",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10,
        help="Chu kỳ lưu checkpoint (mặc định: 10)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Không tải checkpoint cũ, chạy lại từ đầu",
    )
    parser.add_argument(
        "--no-retry-failed",
        action="store_true",
        default=False,
        help="Bỏ qua các câu hỏi từng bị lỗi trong checkpoint (không thử lại)",
    )
    parser.add_argument(
        "--dry-run",
        "--mock",
        action="store_true",
        dest="dry_run",
        default=False,
        help="Chế độ mock chạy thử cục bộ không cần nạp mô hình LLM",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    ckpt_p = Path(args.checkpoint_path) if args.checkpoint_path else None
    if ckpt_p and not ckpt_p.is_absolute():
        ckpt_p = PROJECT_ROOT / ckpt_p

    run_llm_only_experiment(
        config_path=cfg_path,
        device=args.device,
        api_base=args.api_base,
        api_key=args.api_key,
        sample_limit_override=args.sample_limit,
        checkpoint_path=ckpt_p,
        checkpoint_freq=args.checkpoint_freq,
        resume=not args.no_resume,
        retry_failed=not args.no_retry_failed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
