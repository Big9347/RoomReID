# Dockerfile.kafka-milvus
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if needed (e.g., gcc, curl)
RUN apt-get update && apt-get install -y \
    gcc \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for caching
COPY ./analytics/mtmc_analytics/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ./analytics/mtmc_analytics/ .


