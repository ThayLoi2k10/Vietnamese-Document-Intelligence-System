"""Execution orchestrator for zero-context LLM-only baselines."""

import time
from typing import Any, Dict, Optional

from src.rag.abstention.detection.abstention_detector import AbstentionDetector
from src.rag.abstention.detection.models import (
    AbstentionDecision,
    AbstentionReasonCode,
)
from src.rag.abstention.policies.abstention_validator import AbstentionValidator
from src.rag.citation.validation.citation_validator import CitationValidator
from src.rag.citation.validation.models import CitationValidationResult
from src.rag.context.domain.models import AssembledContext
from src.rag.errors.exceptions import InvalidPromptError
from src.rag.generation.domain.models import (
    GenerationParameters,
    GenerationRequest,
)
from src.rag.generation.interfaces.generator import LLMGenerator
from src.rag.output.parsers.structured_parser import OutputParser
from src.rag.output.schemas.response_schema import (
    ConfidenceLevel,
    LegalAnswer,
)
from src.rag.pipeline.models import RAGResponse
from src.rag.prompting.builders.prompt_builder import PromptBuilder


class LLMOnlyPipeline:
    """Orchestrates zero-context LLM inference producing normalized RAGResponse objects."""

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        generator: LLMGenerator,
        output_parser: OutputParser,
        abstention_detector: AbstentionDetector,
        abstention_validator: Optional[AbstentionValidator] = None,
        citation_validator: Optional[CitationValidator] = None,
        default_model_identifier: str = "Qwen/Qwen2.5-7B-Instruct",
        pipeline_version: str = "llm-only-pipeline-v1.0",
    ) -> None:
        self.prompt_builder = prompt_builder
        self.generator = generator
        self.output_parser = output_parser
        self.abstention_detector = abstention_detector
        self.abstention_validator = (
            abstention_validator or AbstentionValidator()
        )
        self.citation_validator = citation_validator or CitationValidator()
        self.default_model_identifier = default_model_identifier
        self.pipeline_version = pipeline_version

    def run(
        self,
        query: str,
        prompt_version: str = "prompt_llm_only_v1",
        model_identifier: Optional[str] = None,
        generation_parameters: Optional[GenerationParameters] = None,
        thinking_mode: bool = False,
        max_output_tokens: int = 1024,
        run_metadata: Optional[Dict[str, Any]] = None,
    ) -> RAGResponse:
        start_time = time.time()

        if not query or not query.strip():
            raise InvalidPromptError("Query cannot be empty.")

        clean_query = query.strip()
        active_model = model_identifier or self.default_model_identifier
        gen_params = generation_parameters or GenerationParameters()

        empty_context = AssembledContext(
            query=clean_query,
            items=[],
            total_tokens=0,
            context_metadata={"system_type": "LLM_ONLY", "retrieval": "DISABLED"},
        )

        prompt_request = self.prompt_builder.build(
            query=clean_query,
            context=empty_context,
            template_version=prompt_version,
        )

        generation_req = GenerationRequest(
            model_identifier=active_model,
            prompt=prompt_request,
            max_output_tokens=max_output_tokens,
            generation_parameters=gen_params,
            thinking_mode=thinking_mode,
            request_metadata=run_metadata or {},
        )
        generation_result = self.generator.generate(generation_req)

        legal_answer: LegalAnswer = self.output_parser.parse(
            generation_result.raw_response
        )

        citation_validation = CitationValidationResult(
            is_valid=True,
            valid_citations=legal_answer.citations,
            invalid_citations=[],
            errors=[],
            quote_matches={},
        )

        if legal_answer.confidence_level == ConfidenceLevel.INSUFFICIENT:
            abstention_decision = AbstentionDecision(
                status=ConfidenceLevel.INSUFFICIENT,
                reason_code=AbstentionReasonCode.MODEL_REPORTED_INSUFFICIENT,
                explanation=(
                    legal_answer.abstention_reason
                    or "Mô hình LLM báo cáo không đủ kiến thức/thông tin để trả lời."
                ),
                supporting_evidence_count=0,
                policy_version=self.pipeline_version,
            )
        elif legal_answer.confidence_level == ConfidenceLevel.PARTIAL:
            abstention_decision = AbstentionDecision(
                status=ConfidenceLevel.PARTIAL,
                reason_code=AbstentionReasonCode.MODEL_REPORTED_PARTIAL,
                explanation=(
                    legal_answer.abstention_reason
                    or "Mô hình LLM báo cáo chỉ trả lời được một phần câu hỏi."
                ),
                supporting_evidence_count=0,
                policy_version=self.pipeline_version,
            )
        else:
            abstention_decision = AbstentionDecision(
                status=ConfidenceLevel.FULL,
                reason_code=AbstentionReasonCode.SUFFICIENT_EVIDENCE,
                explanation="Mô hình LLM báo cáo có đủ kiến thức để trả lời câu hỏi.",
                supporting_evidence_count=0,
                policy_version=self.pipeline_version,
            )

        total_latency = time.time() - start_time

        exec_metadata = {
            "pipeline_version": self.pipeline_version,
            "system_type": "LLM_ONLY",
            "retrieval_items_count": 0,
            "context_items_count": 0,
            "total_latency_seconds": total_latency,
            "model_identifier": active_model,
        }
        if run_metadata:
            exec_metadata.update(run_metadata)

        return RAGResponse(
            question=clean_query,
            legal_answer=legal_answer,
            citation_validation=citation_validation,
            abstention_decision=abstention_decision,
            assembled_context=empty_context,
            generation_result=generation_result,
            execution_metadata=exec_metadata,
        )
