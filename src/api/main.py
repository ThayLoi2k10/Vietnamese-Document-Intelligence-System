"""Vietnamese Legal RAG REST API Entrypoint."""

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.api.routes import get_rag_service, router as api_router

logger = logging.getLogger("legal_rag_api")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle events."""
    import threading

    logger.info("Khởi động FastAPI Legal RAG Service...")

    def bg_load():
        try:
            logger.info("Đang nạp dữ liệu và tải trước các mô hình trong nền...")
            service = get_rag_service()
            service.load()
            logger.info(
                f"Đã nạp LegalRAGService thành công với {len(service.corpus):,} điều luật."
            )
        except Exception as exc:
            logger.warning(f"Chưa thể nạp toàn bộ chỉ mục lúc khởi động: {exc}")

    threading.Thread(target=bg_load, daemon=True).start()

    yield

    logger.info("Đang tắt FastAPI Legal RAG Service...")


def create_app() -> FastAPI:
    """Application factory for FastAPI service."""
    app = FastAPI(
        title="Vietnamese Legal RAG API",
        description=(
            "Production-ready REST API for Vietnamese Statutory Retrieval and Question Answering. "
            "Features Hybrid Search (BM25 + BGE-M3 Dense), Cross-Encoder Reranking, "
            "Passage Compression, and Citation Grounding Verification."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Enable CORS for Streamlit UI or React Frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(api_router)

    # Kubernetes / Docker liveness probe at root
    @app.get("/health", tags=["System"])
    def root_health():
        return {"status": "ok", "service": "Vietnamese Legal RAG API"}

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
