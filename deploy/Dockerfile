# File: deploy/Dockerfile
FROM python:3.10-slim

# 1. Force the app to pick up /app/models via environment
ENV USE_EMBEDDED_MODEL=1

# 2. Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      libsm6 \
      libxext6 && \
    rm -rf /var/lib/apt/lists/*

# 3. Set working dir
WORKDIR /app

# 4. Copy requirements & config from repo root
COPY requirements-runtime.txt config.json /app/

# 5. Copy pretrained weights directory (models/) from repo root
COPY models /app/models

# 6. Copy the FastAPI app from deploy/ subfolder
COPY deploy/app.py /app/app.py

# 7. Install Python deps
RUN pip install --no-cache-dir -r requirements-runtime.txt

# 8. Expose port & run
EXPOSE 8889
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8889"]
