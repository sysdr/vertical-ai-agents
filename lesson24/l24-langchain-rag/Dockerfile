FROM python:3.11-slim

WORKDIR /app

# Optimize pip for long installs (prefer binaries, CPU wheels, no cache)
ARG PIP_NO_CACHE_DIR=1
ARG PIP_PREFER_BINARY=1
ARG PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PIP_NO_CACHE_DIR=${PIP_NO_CACHE_DIR}
ENV PIP_PREFER_BINARY=${PIP_PREFER_BINARY}
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend files
COPY backend/requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY data/ ./data/

# Create directories
RUN mkdir -p data/chroma_db

# GEMINI_API_KEY should be set via environment variable or docker-compose.yml
# ENV GEMINI_API_KEY=

EXPOSE 8000

CMD ["python", "backend/main.py"]
