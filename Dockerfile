FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr for real-time logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required by OpenCV matrix operations and FFmpeg audio stream processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Leverage Docker layer caching by installing dependencies before copying application code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source repository and pre-create persistent artifact & temporary storage directories
COPY . .
RUN mkdir -p /app/artifacts /app/temp_data

EXPOSE 8000

# Execute Uvicorn ASGI server bound to all network interfaces
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]