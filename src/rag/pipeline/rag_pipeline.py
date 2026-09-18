# src/rag/pipeline/rag_pipeline.py
"""Bộ điều phối luồng thực thi chính (RAGPipeline Orchestrator) cho Phase 9.

Tuân thủ Mục 16: RAG Pipeline:
    - 16.1 Orchestrator & Dependency Injection
    - 16.2 Execution flow: Query -> Retrieval -> (Rerank) -> Context -> Prompt -> Generation -> Parse -> Citation -> Abstention -> Response
    - 16.3 Phân tách trách nhiệm rõ ràng, không fallback ngầm
    - 16.4 Kiểm thử cô lập qua Mock
"""

import time
from typing import Any, Dict, Optional

from src.rag.abstention.detection.abstention_detector import AbstentionDetector
from src.rag.abstention.detection.models import (
    AbstentionDecision,
    AbstentionReasonCode,
)
from src.rag.abstention.policies.abstention_validator import AbstentionValidator
from src.rag.citation.validation.citation_validator import CitationValidator
from src.rag.context.context_assembler import ContextAssembler
from src.rag.context.domain.models import AssembledContext
from src.rag.context.policies.policy_types import ContextPolicy
from src.rag.errors.exceptions import InvalidPromptError
from src.rag.generation.domain.models import (
    GenerationParameters,
    GenerationRequest,
)
from src.rag.generation.interfaces.generator import LLMGenerator
from src.rag.output.parsers.structured_parser import OutputParser
from src.rag.output.schemas.response_schema import ConfidenceLevel, LegalAnswer
from src.rag.pipeline.models import RAGResponse
from src.rag.prompting.builders.prompt_builder import PromptBuilder
from src.rag.reranking.domain.models import RerankOptions
from src.rag.reranking.interfaces.reranker import Reranker
from src.rag.retrieval.domain.models import RetrievalOptions
from src.rag.retrieval.interfaces.retriever import Retriever


class RAGPipeline:
    """Bộ điều phối RAG thực thi toàn bộ pipeline từ truy vấn đến câu trả lời pháp lý có cấu trúc."""

    def __init__(
        self,
        retriever: Retriever,
        context_assembler: ContextAssembler,
        prompt_builder: PromptBuilder,
        generator: LLMGenerator,
        output_parser: OutputParser,
        citation_validator: CitationValidator,
        abstention_detector: AbstentionDetector,
        reranker: Optional[Reranker] = None,
        abstention_validator: Optional[AbstentionValidator] = None,
        default_model_identifier: str = "Qwen/Qwen2.5-7B-Instruct",
        pipeline_version: str = "rag-pipeline-v0.9",
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.context_assembler = context_assembler
        self.prompt_builder = prompt_builder
        self.generator = generator
        self.output_parser = output_parser
        self.citation_validator = citation_validator
        self.abstention_detector = abstention_detector
        self.abstention_validator = (
            abstention_validator or AbstentionValidator()
        )
        self.default_model_identifier = default_model_identifier
        self.pipeline_version = pipeline_version

    def run(
        self,
        query: str,
        retrieval_options: Optional[RetrievalOptions] = None,
        rerank_options: Optional[RerankOptions] = None,
        context_policy: Optional[ContextPolicy] = None,
        prompt_version: Optional[str] = None,
        model_identifier: Optional[str] = None,
        generation_parameters: Optional[GenerationParameters] = None,
        thinking_mode: bool = False,
        max_output_tokens: int = 1024,
        run_metadata: Optional[Dict[str, Any]] = None,
    ) -> RAGResponse:
        """Thực thi một chu trình RAG hoàn chỉnh cho câu hỏi truy vấn.

        Args:
            query: Câu hỏi pháp lý của người dùng.
            retrieval_options: Tùy chọn truy xuất (top_k, mode, etc.).
            rerank_options: Tùy chọn reranking (top_k, etc.).
            context_policy: Chính sách lắp ráp ngữ cảnh.
            prompt_version: Phiên bản prompt template.
            model_identifier: Tên mô hình LLM suy luận.
            generation_parameters: Siêu tham số sinh văn bản.
            thinking_mode: Bật/tắt chế độ suy nghĩ của mô hình.
            max_output_tokens: Số token tối đa cho kết quả sinh.
            run_metadata: Metadata kèm theo lượt chạy.

        Returns:
            Đối tượng RAGResponse chuẩn hóa.
        """
        start_time = time.time()

        if not query or not query.strip():
            raise InvalidPromptError("Câu hỏi truy vấn không được để trống.")

        clean_query = query.strip()
        r_opts = retrieval_options or RetrievalOptions(retrieval_top_k=10)
        active_model = model_identifier or self.default_model_identifier
        gen_params = generation_parameters or GenerationParameters()

        # 1. BƯỚC TRUY XUẤT (RETRIEVAL)
        retrieval_result = self.retriever.retrieve(clean_query, r_opts)

        # 2. BƯỚC XẾP HẠNG LẠI (OPTIONAL RERANKING)
        ranked_evidence: Any = retrieval_result
        if self.reranker is not None and not retrieval_result.is_empty:
            rr_opts = rerank_options or RerankOptions(
                reranker_top_k=min(len(retrieval_result.items), 10)
            )
            ranked_evidence = self.reranker.rerank(
                clean_query, retrieval_result.items, rr_opts
            )

        # 3. BƯỚC LẮP RÁP NGỮ CẢNH (CONTEXT ASSEMBLY)
        assembled_context: AssembledContext = self.context_assembler.assemble(
            ranked_evidence,
            policy=context_policy,
            query=clean_query,
        )

        # 4. BƯỚC XÂY DỰNG PROMPT (PROMPT BUILDING)
        prompt_request = self.prompt_builder.build(
            query=clean_query,
            context=assembled_context,
            template_version=prompt_version,
        )

        # 5. BƯỚC SINH VĂN BẢN (LLM GENERATION)
        generation_req = GenerationRequest(
            model_identifier=active_model,
            prompt=prompt_request,
            max_output_tokens=max_output_tokens,
            generation_parameters=gen_params,
            thinking_mode=thinking_mode,
            request_metadata=run_metadata or {},
        )
        generation_result = self.generator.generate(generation_req)

        # 6. BƯỚC PHÂN TÍCH CẤU TRÚC ĐẦU RA (OUTPUT PARSING)
        legal_answer: LegalAnswer = self.output_parser.parse(
            generation_result.raw_response
        )

        # 7. BƯỚC XÁC THỰC TRÍCH DẪN (CITATION VALIDATION)
        citation_validation = self.citation_validator.validate(
            legal_answer,
            assembled_context.evidence_registry,
        )

        # 8. BƯỚC PHÁT HIỆN VÀ ĐIỀU HÒA TỪ CHỐI (ABSTENTION DETECTION & VALIDATION)
        raw_decision: AbstentionDecision = self.abstention_detector.detect(
            context=assembled_context,
            answer=legal_answer,
            citation_validation=citation_validation,
        )
        final_decision: AbstentionDecision = (
            self.abstention_validator.validate_and_reconcile(
                decision=raw_decision,
                answer=legal_answer,
                context=assembled_context,
                citation_validation=citation_validation,
            )
        )

        elapsed_ms = (time.time() - start_time) * 1000.0

        exec_metadata = {
            "pipeline_version": self.pipeline_version,
            "latency_ms": elapsed_ms,
            "retrieval_mode": retrieval_result.retrieval_mode.value,
            "retrieval_items_count": len(retrieval_result.items),
            "context_items_count": len(assembled_context.items),
            "reranker_used": self.reranker is not None,
            "model_identifier": active_model,
            "thinking_mode": thinking_mode,
            "custom_metadata": run_metadata or {},
        }

        return RAGResponse(
            question=clean_query,
            legal_answer=legal_answer,
            citation_validation=citation_validation,
            abstention_decision=final_decision,
            assembled_context=assembled_context,
            generation_result=generation_result,
            execution_metadata=exec_metadata,
        )
