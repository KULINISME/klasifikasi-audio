# ==============================================================================
# Dockerfile: Hening - Real-Time Audio Classification Web Service (Flask + SVM)
# Base Image: Python 3.12 Slim (Debian Bookworm)
# ==============================================================================
FROM python:3.12-slim

# Set metadata
LABEL maintainer="Hening DevOps Team <devops@hening.local>"
LABEL description="Containerized Real-Time Audio Noise Classifier using Flask and SVM"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_RUN_HOST=0.0.0.0 \
    PORT=5000

# Install system dependencies for audio capture (PortAudio, ALSA, libsndfile, FFmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1 \
    alsa-utils \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency definition first for Docker layer caching
COPY requirements.txt /app/

# Upgrade pip and install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application codebase and machine learning model artifacts
COPY . /app/

# Ensure templates and model files have correct permissions
RUN chmod -R 755 /app

# Expose only 1 single port as requested
EXPOSE 5000

# Health check to ensure Flask server is responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/api/status || exit 1

# Start Flask application
CMD ["python", "app.py"]

