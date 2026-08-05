# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# gcc/g++ needed by some sentence-transformers transitive deps on arm64.
# psycopg2-binary and pymupdf ship prebuilt wheels — no libpq-dev/libmupdf-dev needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency manifests first so layer is cached on code-only changes.
COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --upgrade pip && pip install "." --no-cache-dir

# Pre-download the embedding model so it's baked into the image.
# This prevents a slow cold start on first request.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy pre-downloaded model cache
COPY --from=builder /root/.cache /root/.cache

# Copy application source
COPY src/ ./src/
COPY migrations/ ./migrations/
COPY scripts/ ./scripts/
COPY prompts/ ./prompts/
COPY data/ ./data/
COPY alembic.ini .

# Create upload temp directory
RUN mkdir -p /tmp/career_copilot_uploads

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check — used by Docker and Render
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
