"""Pydantic schemas for Vietnamese Legal RAG REST API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievedDocument(BaseModel):
    """Schema for a single retrieved legal article."""

    rank: int = Field(..., description="Rank in retrieval results (1-indexed)")
    article_id: int = Field(..., description="Unique global article ID")
    law_id: str = Field(..., description="Formal legal code (e.g., '45/2019/QH14')")
    aid: str = Field(..., description="Article identifier (e.g., '54231' or 'Điều 35')")
    score: float = Field(..., description="Relevance / similarity score")
    text: str = Field(..., description="Text content of the legal article")


class SearchRequest(BaseModel):
    """Request payload for search endpoint."""

    query: str = Field(..., min_length=1, description="Search query string")
    retriever_type: str = Field(
        default="hybrid_rerank",
        description="Retriever type: 'bm25', 'dense', 'hybrid', or 'hybrid_rerank'",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top legal articles to retrieve",
    )


class SearchResponse(BaseModel):
    """Response payload for search endpoint."""

    query: str = Field(..., description="Original query")
    normalized_query: str = Field(..., description="Query after acronym expansion")
    total_results: int = Field(..., description="Number of articles returned")
    retriever_type: str = Field(..., description="Retriever algorithm used")
    documents: List[RetrievedDocument] = Field(
        default_factory=list, description="Retrieved documents"
    )
    latency_ms: float = Field(..., description="Search latency in milliseconds")


class QueryRequest(BaseModel):
    """Request payload for end-to-end question answering endpoint."""

    query: str = Field(..., min_length=1, description="User legal question")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Prior conversation turns for multi-turn contextualization",
    )
    backend: str = Field(
        default="auto",
        description="LLM backend: 'auto', 'colab', 'groq', 'gemini', 'ollama', 'openai', or 'none'",
    )
    colab_url: Optional[str] = Field(
        default=None,
        description="Cloudflare Tunnel or endpoint URL for Colab GPU server",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for cloud LLM provider (Groq, Gemini, OpenAI)",
    )
    model: Optional[str] = Field(
        default=None,
        description="Specific model name (e.g., 'qwen2.5-7b', 'llama-3.3-70b-versatile')",
    )
    retriever_type: str = Field(
        default="hybrid_rerank",
        description="Retriever type: 'bm25', 'dense', 'hybrid', or 'hybrid_rerank'",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of context documents retrieved for synthesis",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for LLM generation",
    )
    max_tokens: int = Field(
        default=1024,
        ge=64,
        le=4096,
        description="Maximum tokens for generated response",
    )


class GroundingInfo(BaseModel):
    """Schema for citation and grounding verification metrics."""

    score: float = Field(..., description="Grounding score between 0.0 and 1.0")
    verdict: str = Field(..., description="Grounding verdict (VERIFIED, PARTIAL, HALLUCINATION_RISK, ABSTAINED)")
    badge: str = Field(..., description="Display badge string with status emoji")
    summary: str = Field(..., description="Summary explanation of verification findings")
    cited_laws: List[str] = Field(default_factory=list, description="Legal codes cited by LLM")
    cited_articles: List[str] = Field(default_factory=list, description="Articles cited by LLM")
    missing_citations: List[str] = Field(default_factory=list, description="Citations claimed but missing from context")
    ungrounded_citations: List[str] = Field(default_factory=list, description="Ungrounded citations identified")


class QueryResponse(BaseModel):
    """Response payload for end-to-end question answering endpoint."""

    status: str = Field(default="SUCCESS", description="Response status (SUCCESS or ERROR)")
    query: str = Field(..., description="Original user query")
    rewritten_query: Optional[str] = Field(
        default=None, description="Contextualized query if resolved from multi-turn history"
    )
    detected_acronyms: Optional[Dict[str, str]] = Field(
        default=None, description="Acronyms expanded in query"
    )
    answer: str = Field(..., description="Full text response from LLM or structured output")
    direct_conclusion: str = Field(
        default="", description="Section 1: Direct, concise answer"
    )
    legal_grounds: str = Field(
        default="", description="Section 2: Legal bases and citations"
    )
    notes_and_conditions: str = Field(
        default="", description="Section 3: Important conditions and exceptions"
    )
    is_abstention: bool = Field(
        default=False, description="Whether LLM abstained due to insufficient context"
    )
    grounding: Optional[GroundingInfo] = Field(
        default=None, description="Citation and grounding verification details"
    )
    sources: List[RetrievedDocument] = Field(
        default_factory=list, description="Retrieved legal documents used as context"
    )
    latency: Dict[str, float] = Field(
        default_factory=dict, description="Latency breakdown (search_ms, generation_ms, total_ms)"
    )


class DocumentResponse(BaseModel):
    """Response payload for single document lookup."""

    article_id: int = Field(..., description="Global article ID")
    law_id: str = Field(..., description="Legal document code")
    aid: str = Field(..., description="Article identifier")
    text: str = Field(..., description="Full text of the legal article")


class HealthResponse(BaseModel):
    """Response payload for system health check."""

    status: str = Field(default="healthy", description="Overall service status")
    version: str = Field(default="1.0.0", description="API version")
    total_articles: int = Field(..., description="Total legal articles indexed")
    indices_loaded: Dict[str, bool] = Field(
        ..., description="Status of retrieval indices (bm25, dense, reranker)"
    )
    default_backend: str = Field(..., description="Default LLM backend configured")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
