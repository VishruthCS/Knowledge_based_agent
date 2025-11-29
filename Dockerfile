# Use Python 3.11
FROM python:3.11-slim

# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout and stderr (logs show immediately)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
# ADDED 'curl' here so the HEALTHCHECK works
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 1. Copy requirements first (Cached Layer)
COPY requirements.txt .

# 2. Install dependencies (Cached Layer)
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy application code (Frequent Changes)
COPY . .

EXPOSE 8501

# Healthcheck (Now works because we installed curl)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]