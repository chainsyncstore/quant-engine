
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
# We don't have a requirements.txt in root? I'll assume one exists or I should create it.
# For now, I'll install the key packages directly to be safe, or copy a mock one.
# Best practice: Copy source and install via pip
COPY . .

# Install dependencies
RUN pip install --no-cache-dir \
    numpy pandas scikit-learn lightgbm \
    requests python-dotenv \
    sqlalchemy cryptography aiosqlite python-telegram-bot

# Volume for DB and Models
VOLUME ["/app/models", "/app/quant_bot.db"]

CMD ["python", "-m", "quant.telebot.main"]
