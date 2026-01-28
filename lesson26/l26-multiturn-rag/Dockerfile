FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Copy backend
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

# Expose ports
EXPOSE 8000

# Start Redis and backend
CMD redis-server --daemonize yes && \
    uvicorn app.main:app --host 0.0.0.0 --port 8000
