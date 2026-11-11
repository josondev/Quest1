FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Prefer the Ubuntu main archive if an image source points at the Azure mirror.
# Noble can store mirror definitions in /etc/apt/apt-mirrors.txt, so update it
# alongside conventional .list and .sources files before refreshing APT.
RUN find /etc/apt -type f \( -name '*.list' -o -name '*.sources' -o -name 'apt-mirrors.txt' \) -print0 \
    | xargs -r -0 sed -i 's|azure.archive.ubuntu.com|archive.ubuntu.com|g' \
    && apt-get update -o Acquire::Retries=3 \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        curl \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Deno (required by yt-dlp EJS for YouTube JS challenge solving)
RUN curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh
ENV PATH="/usr/local/bin:${PATH}"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/artifacts /app/temp_data

RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
