# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps: ffmpeg for PyAV; libgomp1 for onnxruntime; portaudio for sounddevice (safe even if unused)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg ca-certificates libgomp1 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port for health checks (optional)
EXPOSE 8080

# Background worker entrypoint (no HTTP)
# Use 'start' for production, 'dev' for local development
RUN pip install --upgrade pip

# Copy and install Python dependencies
RUN pip install -r requirements.txt


# Start Gunicorn to serve the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]