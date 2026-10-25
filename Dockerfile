# ==============================================================================
# Dockerfile — Vietnamese Legal Document Intelligence & RAG System
# Chạy đầy đủ: Hybrid (BM25 + BGE-M3) + Fine-tuned LoRA Legal Reranker + Streamlit UI
# ==============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài đặt dependencies (bao gồm PyTorch CPU, Transformers, FAISS, Streamlit)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy cấu hình, mã nguồn và trọng số LoRA đã fine-tune
COPY .streamlit/ /app/.streamlit/
COPY configs/ /app/configs/
COPY src/ /app/src/
COPY data/processed/legal_retrieval/corpus_articles_preprocessed.parquet /app/data/processed/legal_retrieval/corpus_articles_preprocessed.parquet
COPY data/indices/bm25_index.pkl /app/data/indices/bm25_index.pkl
COPY data/indices/hybrid_config.json /app/data/indices/hybrid_config.json
COPY data/indices/reranker_finetuned/ /app/data/indices/reranker_finetuned/
COPY app.py /app/app.py

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
