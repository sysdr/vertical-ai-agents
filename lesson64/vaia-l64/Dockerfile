# =============================================================================
# L64 VAIA Agent – Multi-stage Dockerfile
# Stage 1 (builder): install all deps into a venv
# Stage 2 (dev):     adds test deps – used by docker compose test service
# Stage 3 (runtime): minimal production image
# =============================================================================

# ──────────────────────────────────────────────────────────────
# Stage 1: builder
# ──────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Layer-cache optimisation: copy deps file first
COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc curl \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /app/.venv \
    && /app/.venv/bin/pip install --no-cache-dir --upgrade pip \
    && /app/.venv/bin/pip install --no-cache-dir -r requirements.txt


# ──────────────────────────────────────────────────────────────
# Stage 2: dev  (assignment target – adds pytest)
# ──────────────────────────────────────────────────────────────
FROM builder AS dev

COPY requirements-dev.txt .
RUN /app/.venv/bin/pip install --no-cache-dir -r requirements-dev.txt

# Application source is volume-mounted in Compose for fast iteration
COPY src/ src/
COPY tests/ tests/

ENV PATH="/app/.venv/bin:$PATH"
CMD ["pytest", "tests/", "-v", "--tb=short"]


# ──────────────────────────────────────────────────────────────
# Stage 3: runtime  (production)
# ──────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="vaia-agent"
LABEL org.opencontainers.image.description="VAIA Curriculum L64 – Containerised agent"
LABEL org.opencontainers.image.version="0.64.0"

WORKDIR /app

# Only copy the virtual env and source – no build tools in final image
COPY --from=builder /app/.venv /app/.venv
COPY src/ src/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Docker-native liveness – K8s probes declared in deployment.yaml
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Non-root user for security hardening
RUN useradd -r -u 1001 -g root vaia
USER vaia

CMD ["uvicorn", "src.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--loop", "uvloop"]
