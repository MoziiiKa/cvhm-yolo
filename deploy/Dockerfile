# File: deploy/Dockerfile
FROM python:3.10-slim

# 1. Ensure the environment variable tells app.py to load /app/models
ENV USE_EMBEDDED_MODEL=1

# 2. Install system dependencies for video/image processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      libsm6 \
      libxext6 && \
    rm -rf /var/lib/apt/lists/*

# 3. Set working directory inside the container
WORKDIR /app

# 4. Copy Python requirements and project configuration
COPY requirements.txt config.json /app/

# 5. Copy your pretrained weights into /app/models
#    (assumes 'models/best.pt' exists in the build context)
COPY ../models /app/models

# 6. Copy the FastAPI inference app
COPY app.py /app/

# 7. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 8. Expose the HTTP port and default startup command
EXPOSE 8889
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8889"]
