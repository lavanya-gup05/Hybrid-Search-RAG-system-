# Slim Python base. sentence-transformers/torch make this a fairly heavy
# image regardless of base — no way around that for a self-hosted embedding
# model — but slim keeps the OS layer small.
FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffers, and keeps pip quiet.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps: build tools for any package that needs to compile (rank-bm25
# and some torch wheels pull these in on certain platforms), plus curl for
# the healthcheck below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install deps first, separately from app code, so `docker build` reuses this
# (slow) layer on every rebuild that only touches source files.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Where Chroma persists its index and where sentence-transformers caches
# downloaded model weights — both mounted as a volume in docker-compose so
# neither is re-downloaded / re-embedded on every container restart.
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/index /app/.cache/huggingface

# Don't run the app as root inside the container: create a dedicated user and
# hand it ownership of the writable paths (app dir, index, model cache, and
# /tmp, which the app uses for per-session upload and index directories).
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app /tmp
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", "--server.address=0.0.0.0"]