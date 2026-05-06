FROM python:3.12-slim

WORKDIR /app

# curl required for Docker Compose healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN addgroup --system vaia && adduser --system --group vaia

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent_service/ ./agent_service/
COPY monitoring/ ./monitoring/
COPY pipeline/ ./pipeline/

RUN mkdir -p data && chown -R vaia:vaia /app
USER vaia

# Cloud Run / Kubernetes lifecycle probes
EXPOSE 8080
ENV PORT=8080

# Graceful shutdown: uvicorn honours SIGTERM with --timeout-graceful-shutdown
CMD ["sh", "-c", "uvicorn agent_service.main:app --host 0.0.0.0 --port ${PORT} --timeout-graceful-shutdown 30"]
