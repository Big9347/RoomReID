# Dockerfiles/flask-webui.Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if needed (e.g., gcc, curl)
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for caching
COPY ./analytics/mtmc_analytics/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy both analytics components relative to the root project
COPY ./analytics/ ./

# Expose the Flask Port
EXPOSE 5000

# Switch context to the web application
WORKDIR /app/web_ui

# Set entry point
CMD ["python", "web_app.py"]
