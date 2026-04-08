# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage Dockerfile for the Smart Post-Purchase Support Triage System
# Packages FastAPI backend, Streamlit frontend, and SQLite database
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies Stage ───────────────────────────────────────────────────────

FROM base AS dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application Stage ────────────────────────────────────────────────────────

FROM dependencies AS application

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY training/ ./training/
COPY models/ ./models/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p /app/models /app/mlruns

# ── Training Stage (optional — build with --target=train) ────────────────────

FROM application AS train

RUN python training/train.py

# ── Production Stage ─────────────────────────────────────────────────────────

FROM application AS production

# Expose ports: FastAPI (8000) + Streamlit (8501)
EXPOSE 8000 8501

# Create startup script
RUN echo '#!/bin/bash\n\
echo "🚀 Starting Smart Post-Purchase Support Triage System..."\n\
echo ""\n\
echo "📦 Training model if not present..."\n\
if [ ! -f /app/models/intent_classifier.joblib ]; then\n\
    python training/train.py\n\
fi\n\
echo ""\n\
echo "🌐 Starting FastAPI backend on port 8000..."\n\
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &\n\
echo "📊 Starting Streamlit dashboard on port 8501..."\n\
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true\n\
' > /app/start.sh && chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/start.sh"]
