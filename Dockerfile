
# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY quant/ ./quant/
COPY quant_v2/ ./quant_v2/

RUN pip install --no-cache-dir --prefix=/install \
    numpy pandas scikit-learn lightgbm \
    requests python-dotenv sqlalchemy cryptography aiosqlite \
    python-telegram-bot "redis[async]"

# ---- Stage 2: Runtime (no compiler, no root) ----
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy only application code (no build tools, no .pem, no debug scripts)
COPY quant/ ./quant/
COPY quant_v2/ ./quant_v2/
COPY pyproject.toml ./

# Create non-root user
RUN groupadd -r quantbot && useradd -r -g quantbot -s /sbin/nologin quantbot \
    && chown -R quantbot:quantbot /app

USER quantbot

CMD ["python", "-m", "quant_v2.execution.main"]
