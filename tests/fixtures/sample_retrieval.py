# tests/fixtures/sample_retrieval.py
"""Dữ liệu giả định phục vụ kiểm thử cho Phase 9 RAG."""

from src.rag.context.domain.models import ContextItem
from src.rag.output.schemas.response_schema import Citation, ConfidenceLevel, LegalAnswer
from src.rag.retrieval.domain.models import (
    RetrievalItem,
    RetrievalMetadata,
    RetrievalMode,
    RetrievalResult,
)


def make_sample_retrieval_items() -> list[RetrievalItem]:
    """Tạo danh sách mẫu các RetrievalItem chuẩn."""
    return [
        RetrievalItem(
            chunk_id="chunk_001",
            global_article_id=101,
            law_id="100/2015/NĐ-CP",
            article_identifier="Điều 5",
            text="Người điều khiển phương tiện tham gia giao thông phải chấp hành hiệu lệnh của đèn tín hiệu giao thông.",
            chunk_order=1,
            total_chunks=2,
            score=0.88,
            rank=1,
            token_count=25,
            source_metadata={"database_id": "db_001", "internal_faiss_idx": 999},
        ),
        RetrievalItem(
            chunk_id="chunk_002",
            global_article_id=101,
            law_id="100/2015/NĐ-CP",
            article_identifier="Điều 5",
            text="Trường hợp có người điều khiển giao thông thì phải chấp hành hiệu lệnh của người điều khiển giao thông.",
            chunk_order=2,
            total_chunks=2,
            score=0.75,
            rank=2,
            token_count=26,
            source_metadata={"database_id": "db_002", "internal_faiss_idx": 1000},
        ),
        RetrievalItem(
            chunk_id="chunk_003",
            global_article_id=102,
            law_id="100/2015/NĐ-CP",
            article_identifier="Điều 6",
            text="Xử phạt người điều khiển xe mô tô, xe gắn máy không chấp hành hiệu lệnh đèn tín hiệu.",
            chunk_order=1,
            total_chunks=1,
            score=0.65,
            rank=3,
            token_count=22,
            source_metadata={"database_id": "db_003", "internal_faiss_idx": 1001},
        ),
    ]


def make_sample_retrieval_result() -> RetrievalResult:
    """Tạo đối tượng RetrievalResult mẫu hoàn chỉnh."""
    items = make_sample_retrieval_items()
    return RetrievalResult(
        query="Không chấp hành hiệu lệnh của cảnh sát giao thông thì xử phạt thế nào?",
        retrieval_mode=RetrievalMode.CHUNK,
        items=items,
        retriever_metadata=RetrievalMetadata(
            retriever_name="mock_vector_store",
            retrieval_mode=RetrievalMode.CHUNK,
            top_k=3,
            component_version="v1.0",
        ),
    )


def make_sample_context_items() -> list[ContextItem]:
    """Tạo danh sách ContextItem mẫu."""
    return [
        ContextItem(
            context_item_id="[1]",
            source_identity="chunk_001",
            law_id="100/2015/NĐ-CP",
            article_identifier="Điều 5",
            text="Người điều khiển phương tiện tham gia giao thông phải chấp hành hiệu lệnh của đèn tín hiệu giao thông.",
            source_rank=1,
            context_position=1,
            chunk_id="chunk_001",
            global_article_id=101,
            source_score=0.88,
            token_count=25,
            internal_metadata={"internal_vector_score": 0.88},
        ),
        ContextItem(
            context_item_id="[2]",
            source_identity="chunk_002",
            law_id="100/2015/NĐ-CP",
            article_identifier="Điều 5",
            text="Trường hợp có người điều khiển giao thông thì phải chấp hành hiệu lệnh của người điều khiển giao thông.",
            source_rank=2,
            context_position=2,
            chunk_id="chunk_002",
            global_article_id=101,
            source_score=0.75,
            token_count=26,
            internal_metadata={"internal_vector_score": 0.75},
        ),
    ]


def make_sample_legal_answer() -> LegalAnswer:
    """Tạo câu trả lời pháp lý mẫu hợp lệ."""
    return LegalAnswer(
        answer="Người tham gia giao thông phải chấp hành hiệu lệnh của người điều khiển giao thông ngay cả khi có đèn tín hiệu.",
        citations=[
            Citation(
                citation_number=2,
                law_id="100/2015/NĐ-CP",
                article_identifier="Điều 5",
                chunk_id="chunk_002",
                global_article_id=101,
                supporting_evidence_or_quote="phải chấp hành hiệu lệnh của người điều khiển giao thông",
            )
        ],
        confidence_level=ConfidenceLevel.FULL,
        abstention_reason=None,
    )
