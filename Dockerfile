
# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

# Core dependencies + PyTorch/Chronos for ensemble inference
RUN pip install --no-cache-dir --prefix=/install \
    numpy pandas scikit-learn lightgbm \
    requests python-dotenv sqlalchemy cryptography aiosqlite \
    python-telegram-bot "redis[async]" \
    joblib \
    chronos-forecasting \
    torch --extra-index-url https://download.pytorch.org/whl/cpu \
    typing_extensions

COPY quant/ ./quant/
COPY quant_v2/ ./quant_v2/

# ---- Stage 2: Runtime (no compiler, no root) ----
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy only application code (no build tools, no .pem, no debug scripts)
COPY quant/ ./quant/
COPY quant_v2/ ./quant_v2/
COPY bootstrap_registry.py ./
COPY pyproject.toml ./

# Create non-root user with writable home for model cache
RUN groupadd -r quantbot && useradd -r -g quantbot -s /sbin/nologin -m quantbot \
    && chown -R quantbot:quantbot /app

ENV HF_HOME=/home/quantbot/.cache/huggingface

USER quantbot

CMD ["python", "-m", "quant_v2.execution.main"]
