# scripts/evaluation/run_rag_experiment.py
"""Runner chính thức thực thi thực nghiệm RAG Phase 9 trên Benchmark.

Tuân thủ Mục 16: RAG Pipeline, Mục 17: Evaluation Architecture, Mục 18: Experiment Configuration,
Mục 19: Dataset / Test Protection, và Mục 21: Reproducibility.

Pipeline chuỗi thành phần ghép nối:
    Phase8RetrievalAdapter -> ContextAssembler -> PromptBuilder -> GenericLLMAdapter ->
    OutputParser -> CitationValidator -> AbstentionValidator -> Evaluation
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import urllib.error
import urllib.request

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
from src.rag.abstention.detection.models import (
    AbstentionDecision,
    AbstentionReasonCode,
)
from src.rag.abstention.policies.abstention_validator import AbstentionValidator
from src.rag.citation.validation.citation_validator import CitationValidator
from src.rag.citation.validation.models import CitationValidationResult
from src.rag.config.loaders.config_loader import ConfigLoader
from src.rag.config.schemas.config_models import ExperimentConfig, SplitRole
from src.rag.context.context_assembler import ContextAssembler
from src.rag.context.domain.models import AssembledContext, ContextItem
from src.rag.context.policies.policy_types import (
    BudgetOverflowPolicy,
    ContextPolicy,
    DeduplicationPolicy,
    ExpansionPolicy,
    OrderingStrategy,
)
from src.rag.errors.exceptions import (
    LLMGenerationError,
    MalformedStructuredOutputError,
)
from src.rag.evaluation.backends.rule_based_evaluator import (
    HeuristicFaithfulnessEvaluator,
    RuleBasedAbstentionCorrectnessEvaluator,
    RuleBasedCitationPrecisionEvaluator,
    RuleBasedCitationRecallEvaluator,
)
from src.rag.evaluation.interfaces.evaluator import RAGEvaluator
from src.rag.evaluation.metrics.models import EvaluationInput, EvaluationReport
from src.rag.generation.adapters.generic_llm_adapter import GenericLLMAdapter
from src.rag.generation.domain.models import (
    FinishStatus,
    GenerationParameters,
    GenerationResult,
)
from src.rag.output.parsers.structured_parser import OutputParser
from src.rag.output.schemas.response_schema import (
    Citation,
    ConfidenceLevel,
    LegalAnswer,
)
from src.rag.pipeline.models import RAGResponse
from src.rag.pipeline.rag_pipeline import RAGPipeline
from src.rag.prompting.builders.prompt_builder import PromptBuilder
from src.rag.reproducibility.manifest import ExperimentManifest
from src.rag.retrieval.adapters.phase8_adapter import Phase8RetrievalAdapter
from src.rag.retrieval.domain.models import RetrievalOptions


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
    """Nạp checkpoint an toàn, hỗ trợ cả định dạng dict và list cũ."""
    if not checkpoint_path.exists():
        return [], []
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            # Tương thích ngược nếu checkpoint lưu dạng list các predictions
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
    """Lưu checkpoint nguyên tử để tránh hỏng dữ liệu khi tiến trình bị ngắt đột ngột."""
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


def reconstruct_rag_response_from_dict(d: Dict[str, Any]) -> RAGResponse:
    """Tái tạo đối tượng RAGResponse từ bản ghi audit dict đã lưu trong checkpoint."""
    la_data = d.get("legal_answer", {})
    citations = []
    for c in la_data.get("citations", []):
        if isinstance(c, dict):
            citations.append(Citation.from_dict(c))
        elif isinstance(c, Citation):
            citations.append(c)

    conf_str = str(la_data.get("confidence_level", "FULL")).strip().upper()
    try:
        confidence_level = ConfidenceLevel(conf_str)
    except ValueError:
        confidence_level = ConfidenceLevel.FULL

    legal_answer = LegalAnswer(
        answer=str(la_data.get("answer", "")),
        citations=citations,
        confidence_level=confidence_level,
        abstention_reason=la_data.get("abstention_reason"),
    )

    ctx_data = d.get("assembled_context", {})
    ctx_items = []
    for it in ctx_data.get("items", []):
        ctx_items.append(
            ContextItem(
                context_item_id=it.get("context_item_id", ""),
                source_identity=it.get("source_identity", ""),
                law_id=it.get("law_id", ""),
                text=it.get("text", ""),
                source_rank=it.get("source_rank", 1),
                context_position=it.get("context_position", 1),
                article_identifier=it.get("article_identifier"),
                global_article_id=it.get("global_article_id"),
                chunk_id=it.get("chunk_id"),
                source_score=it.get("source_score"),
                token_count=it.get("token_count"),
                internal_metadata=it.get("internal_metadata", {}),
            )
        )
    assembled_context = AssembledContext(
        query=ctx_data.get("query", d.get("question", "")),
        items=ctx_items,
        total_tokens=ctx_data.get("total_tokens", 0),
        context_metadata=ctx_data.get("context_metadata", {}),
    )

    abs_data = d.get("abstention_decision", {})
    abs_status_str = str(abs_data.get("status", "FULL")).strip().upper()
    try:
        abs_status = ConfidenceLevel(abs_status_str)
    except ValueError:
        abs_status = ConfidenceLevel.FULL
    abs_code_str = str(
        abs_data.get("reason_code", "SUFFICIENT_EVIDENCE")
    ).strip().upper()
    try:
        reason_code = AbstentionReasonCode(abs_code_str)
    except ValueError:
        reason_code = AbstentionReasonCode.SUFFICIENT_EVIDENCE

    abstention_decision = AbstentionDecision(
        status=abs_status,
        reason_code=reason_code,
        explanation=abs_data.get("explanation", ""),
        supporting_evidence_count=abs_data.get("supporting_evidence_count", 0),
        policy_version=abs_data.get("policy_version", ""),
    )

    cv_data = d.get("citation_validation", {})
    citation_validation = CitationValidationResult(
        is_valid=cv_data.get("is_valid", True),
        valid_citations=citations if cv_data.get("is_valid", True) else [],
        invalid_citations=cv_data.get("invalid_citations", []),
        errors=cv_data.get("errors", []),
        quote_matches=cv_data.get("quote_matches", {}),
    )

    gen_data = d.get("generation_result", {})
    fin_status_str = str(
        gen_data.get("finish_status", "COMPLETED")
    ).strip().upper()
    try:
        finish_status = FinishStatus(fin_status_str)
    except ValueError:
        finish_status = FinishStatus.COMPLETED

    generation_result = GenerationResult(
        model_identifier=gen_data.get("model_identifier", ""),
        raw_response=gen_data.get("raw_response", ""),
        generation_metadata=gen_data.get("generation_metadata", {}),
        usage_metadata=gen_data.get("usage_metadata", {}),
        finish_status=finish_status,
    )

    return RAGResponse(
        question=d.get("question", ""),
        legal_answer=legal_answer,
        citation_validation=citation_validation,
        abstention_decision=abstention_decision,
        assembled_context=assembled_context,
        generation_result=generation_result,
        execution_metadata=d.get("execution_metadata", {}),
    )


def build_real_llm_inference_fn(
    model_name: str,
    device: str = "auto",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Callable[[str, str, Dict[str, Any]], str]:
    """Tạo hàm inference_fn thực tế cho GenericLLMAdapter hỗ trợ mô hình chỉ định (ví dụ Qwen2.5-7B).

    Hỗ trợ 2 chế độ:
        1. API endpoint (OpenAI / vLLM compatible) nếu cấu hình api_base hoặc biến môi trường.
        2. Local HuggingFace Transformers (tải trọng số cục bộ khi thực thi script).
    """
    effective_api_base = (
        api_base
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("VLLM_BASE_URL")
        or os.getenv("VLLM_API_BASE")
    )

    if effective_api_base:
        # Chế độ gọi qua HTTP server tương thích vLLM / OpenAI API
        clean_base = effective_api_base.rstrip("/")
        if not clean_base.endswith("/v1"):
            endpoint = f"{clean_base}/v1/chat/completions"
        else:
            endpoint = f"{clean_base}/chat/completions"

        key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"

        def api_inference_fn(
            system_prompt: str, user_prompt: str, call_params: Dict[str, Any]
        ) -> str:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": call_params.get("max_tokens", 1024),
                "temperature": call_params.get("temperature", 0.0),
                "top_p": call_params.get("top_p", 1.0),
            }
            seed = call_params.get("seed")
            if seed is not None:
                payload["seed"] = seed
            rep_penalty = call_params.get("repetition_penalty", 1.1)
            payload["repetition_penalty"] = float(rep_penalty)

            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                endpoint,
                data=req_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                return res_json["choices"][0]["message"]["content"]

        return api_inference_fn

    else:
        # Chế độ chạy mô hình cục bộ qua HuggingFace Transformers (Lazy initialization)
        model_holder: Dict[str, Any] = {"model": None, "tokenizer": None}

        def init_hf_model():
            if model_holder["model"] is None:
                import torch
                from transformers import (
                    AutoModelForCausalLM,
                    AutoTokenizer,
                    BitsAndBytesConfig,
                )

                print(f"[Runner] Nạp mô hình LLM từ HuggingFace: {model_name}...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True
                )
                torch_dtype = (
                    torch.bfloat16
                    if torch.cuda.is_available()
                    and torch.cuda.is_bf16_supported()
                    else torch.float16
                )
                dev_map = "auto" if device == "auto" else device
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=dev_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                )
                model_holder["model"] = model
                model_holder["tokenizer"] = tokenizer
            return model_holder["model"], model_holder["tokenizer"]

        def hf_inference_fn(
            system_prompt: str, user_prompt: str, call_params: Dict[str, Any]
        ) -> str:
            model, tokenizer = init_hf_model()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer([prompt_text], return_tensors="pt").to(
                model.device
            )

            max_new = call_params.get("max_tokens", 1024)
            temp = call_params.get("temperature", 0.0)
            do_sample = temp > 0.0
            rep_penalty = call_params.get("repetition_penalty", 1.1)

            generate_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_new,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.eos_token_id,
                "repetition_penalty": float(rep_penalty),
            }
            if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
                generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
            if do_sample:
                generate_kwargs["temperature"] = temp
                generate_kwargs["top_p"] = call_params.get("top_p", 1.0)

            outputs = model.generate(**inputs, **generate_kwargs)
            generated_ids = outputs[0][inputs.input_ids.shape[1] :]
            return tokenizer.decode(generated_ids, skip_special_tokens=True)

        return hf_inference_fn


def load_dataset_split(
    config: ExperimentConfig, sample_limit_override: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Nạp tập dữ liệu theo đúng vai trò split và sample_limit."""
    split_path = PROJECT_ROOT / config.dataset.split_path
    if not split_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy tệp dữ liệu benchmark: {split_path}"
        )

    with open(split_path, "r", encoding="utf-8") as f:
        all_queries = json.load(f)

    # Phân tách split theo chuẩn Phase 9 (Mục 19.1)
    if config.dataset.split_role == SplitRole.DEVELOPMENT:
        # 1,401 queries đầu tiên của training set
        selected_queries = all_queries[:1401]
    elif config.dataset.split_role == SplitRole.VALIDATION:
        # 351 queries tiếp theo (1,401 -> 1,752)
        selected_queries = all_queries[1401:1752]
    elif config.dataset.split_role == SplitRole.FROZEN_TEST:
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


def run_experiment(
    config_path: Path,
    storage_path: Optional[Path] = None,
    store_type: str = "qdrant",
    device: str = "auto",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    sample_limit_override: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_freq: int = 10,
    resume: bool = True,
    output_dir: Path = PROJECT_ROOT / "results" / "benchmarks",
    artifacts_dir: Path = PROJECT_ROOT / "artifacts",
) -> None:
    """Thực thi toàn bộ quy trình experiment runner theo cấu hình."""
    print("=" * 80)
    print(f"BẮT ĐẦU CHẠY THỰC NGHIỆM RAG PHASE 9: {config_path}")
    print("=" * 80)

    # 1. Nạp và kiểm định cấu hình
    config: ExperimentConfig = ConfigLoader.from_file(
        config_path, validate=True
    )
    print(f"[1/7] Nạp cấu hình thành công. Experiment ID: {config.experiment_id}")
    print(f"      - Dataset Split Role: {config.dataset.split_role.value}")
    print(f"      - LLM Model: {config.generation.llm_model}")
    print(f"      - Retriever: {config.retriever.retriever_name} (top_k={config.retriever.retrieval_top_k})")
    print(f"      - Context top_k: {config.context.context_top_k}")

    # 2. Khởi tạo Phase8RetrievalAdapter với vector store và dense encoder thật
    print("\n[2/7] Khởi tạo Phase8RetrievalAdapter...")
    from src.retrieval.dense.encoder import DenseEncoder

    encoder = DenseEncoder(device=None if device == "auto" else device)

    default_store_path = (
        storage_path
        if storage_path
        else PROJECT_ROOT / "data" / "indices" / "qdrant_legal"
    )

    if store_type.lower() == "qdrant":
        from src.retrieval.indexing.qdrant_store import QdrantVectorStore

        vector_store = QdrantVectorStore(storage_path=default_store_path)
    else:
        from src.retrieval.indexing.faiss_vector_store import FaissVectorStore

        vector_store = FaissVectorStore.load(default_store_path)

    retriever = Phase8RetrievalAdapter(
        vector_store=vector_store,
        encoder=encoder,
        retriever_name=config.retriever.retriever_name,
        component_version=config.retriever.component_version,
    )

    # 3. Khởi tạo ContextAssembler và PromptBuilder
    print("[3/7] Khởi tạo ContextAssembler và PromptBuilder...")
    context_policy = ContextPolicy(
        context_top_k=config.context.context_top_k,
        score_threshold=config.context.score_threshold,
        max_context_items=config.context.max_context_items,
        deduplication_policy=DeduplicationPolicy(
            config.context.deduplication_policy
        ),
        deduplication_options=config.context.deduplication_options,
        expansion_policy=ExpansionPolicy(config.context.expansion_policy),
        expansion_options=config.context.expansion_options,
        ordering_strategy=OrderingStrategy(config.context.ordering_strategy),
        max_context_tokens=config.context.max_context_tokens,
        budget_overflow_policy=BudgetOverflowPolicy(
            config.context.budget_overflow_policy
        ),
        component_version=config.context.component_version,
    )
    context_assembler = ContextAssembler(default_policy=context_policy)
    prompt_builder = PromptBuilder(
        default_version=config.prompting.prompt_version
    )

    # 4. Khởi tạo GenericLLMAdapter với inference function thực tế
    print(
        f"[4/7] Khởi tạo GenericLLMAdapter cho mô hình {config.generation.llm_model}..."
    )
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

    # 5. Khởi tạo Parser, Validators và ghép RAGPipeline
    print("[5/7] Ghép nối các component vào RAGPipeline...")
    output_parser = OutputParser()
    citation_validator = CitationValidator()
    abstention_detector = AbstentionDetector()
    abstention_validator = AbstentionValidator()

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=None,
        context_assembler=context_assembler,
        prompt_builder=prompt_builder,
        generator=generator,
        output_parser=output_parser,
        citation_validator=citation_validator,
        abstention_detector=abstention_detector,
        abstention_validator=abstention_validator,
        default_model_identifier=config.generation.llm_model,
        pipeline_version="rag-pipeline-v0.9",
    )

    # 6. Nạp tập dữ liệu và chạy suy luận có cơ chế checkpoint/resume và cô lập lỗi
    print("\n[6/7] Nạp dataset và thực thi truy vấn...")
    queries = load_dataset_split(
        config, sample_limit_override=sample_limit_override
    )
    print(
        f"      Số lượng mẫu thực thi: {len(queries)} câu hỏi ({config.dataset.split_role.value})"
    )

    # Khởi tạo hoặc nạp checkpoint
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
            f"      [RESUME] Đã tải {len(predictions_record)} kết quả hoàn thành (COMPLETED), "
            f"{len(failures_record)} bản ghi lỗi (FAILED)."
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

        # Kiểm tra trạng thái trong checkpoint
        if q_id in completed_ids:
            print(f"  -> [{idx}/{len(queries)}] Query ID {q_id}: [BỎ QUA - ĐÃ HOÀN THÀNH (COMPLETED)]")
            continue
        if q_id in failed_ids:
            print(f"  -> [{idx}/{len(queries)}] Query ID {q_id}: [BỎ QUA - ĐÃ GHI NHẬN LỖI (FAILED)]")
            continue

        print(f"  -> [{idx}/{len(queries)}] Query ID {q_id}: {q_text[:70]}...")

        try:
            response: RAGResponse = pipeline.run(
                query=q_text,
                retrieval_options=RetrievalOptions(
                    retrieval_top_k=config.retriever.retrieval_top_k
                ),
                context_policy=context_policy,
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

            # Lưu vết prediction
            pred_item = {
                "query_id": q_id,
                "query": q_text,
                "status": "COMPLETED",
                "response": response.to_full_audit_dict(),
            }
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
            if raw_output is not None:
                snippet = (
                    repr(raw_output[:200])
                    if len(raw_output) > 200
                    else repr(raw_output)
                )
                print(f"    Raw output (độ dài {len(raw_output)} chars): {snippet}")

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
            failures_record.append(failure_item)
            failed_ids.add(q_id)
            processed_in_this_run += 1

            # Lưu checkpoint ngay lập tức khi gặp lỗi đầu ra để không mất dấu vết
            save_checkpoint(
                checkpoint_path=default_checkpoint_path,
                experiment_id=config.experiment_id,
                predictions=predictions_record,
                failures=failures_record,
            )

        # Lưu checkpoint định kỳ theo chu kỳ checkpoint_freq
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
                f"      [CHECKPOINT] Đã lưu tiến độ: {len(predictions_record)} hoàn thành, "
                f"{len(failures_record)} lỗi -> {default_checkpoint_path}"
            )

    # Lưu checkpoint cuối cùng sau khi hoàn tất vòng lặp
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

    # 7. Thực hiện đánh giá metrics và lưu trữ Artifacts
    print("\n[7/7] Đánh giá metrics và lưu trữ kết quả...")
    evaluators_map = {
        "citation_precision": RuleBasedCitationPrecisionEvaluator(),
        "citation_recall": RuleBasedCitationRecallEvaluator(),
        "abstention_correctness": RuleBasedAbstentionCorrectnessEvaluator(),
        "faithfulness": HeuristicFaithfulnessEvaluator(),
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
            "retriever": config.retriever.component_version,
            "context": config.context.component_version,
            "prompt": config.prompting.instruction_policy_version,
            "generator": config.generation.component_version,
        },
    )

    # Bổ sung thông tin lỗi và tổng số câu hỏi vào report mà không làm giảm mẫu số benchmark
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

    # Lưu kết quả, predictions, failures và report
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

    # Lưu ExperimentManifest tái lập
    manifest = ExperimentManifest.from_config(config)
    manifest.save(manifest_path)

    print("\n" + "=" * 80)
    print("BÁO CÁO KẾT QUẢ THỰC NGHIỆM:")
    print("=" * 80)
    for m_name, m_val in report.metric_results.items():
        print(f"  * {m_name:<28}: {m_val:.4f}")
    print("-" * 80)
    print(f"Metrics:       {metrics_path}")
    print(f"Predictions:   {predictions_path}")
    print(f"Failures:      {failures_path}")
    print(f"Report:        {report_path}")
    print(f"Manifest:      {manifest_path}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runner thực thi thực nghiệm RAG Phase 9 trên Benchmark"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/baseline_experiment.json",
        help="Đường dẫn file cấu hình thực nghiệm JSON/YAML",
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default=None,
        help="Đường dẫn thư mục lưu trữ Qdrant hoặc FAISS index",
    )
    parser.add_argument(
        "--store-type",
        type=str,
        default="qdrant",
        choices=["qdrant", "faiss"],
        help="Loại kho vector store (mặc định: qdrant)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Thiết bị tính toán cho encoder và local LLM (auto/cuda/cpu)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Base URL nếu sử dụng vLLM / OpenAI-compatible API server",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key nếu sử dụng API server",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Ghi đè số lượng mẫu câu hỏi cần chạy thử nghiệm",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Đường dẫn file checkpoint JSON (mặc định: results/benchmarks/checkpoints/<experiment_id>_checkpoint.json)",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10,
        help="Chu kỳ lưu checkpoint (số lượng query, mặc định: 10)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Không tải checkpoint cũ, chạy lại từ đầu",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    storage_p = Path(args.storage_path) if args.storage_path else None
    if storage_p and not storage_p.is_absolute():
        storage_p = PROJECT_ROOT / storage_p

    ckpt_p = Path(args.checkpoint_path) if args.checkpoint_path else None
    if ckpt_p and not ckpt_p.is_absolute():
        ckpt_p = PROJECT_ROOT / ckpt_p

    run_experiment(
        config_path=cfg_path,
        storage_path=storage_p,
        store_type=args.store_type,
        device=args.device,
        api_base=args.api_base,
        api_key=args.api_key,
        sample_limit_override=args.sample_limit,
        checkpoint_path=ckpt_p,
        checkpoint_freq=args.checkpoint_freq,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
