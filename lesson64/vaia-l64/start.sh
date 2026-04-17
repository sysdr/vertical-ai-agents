#!/usr/bin/env bash
# Start the full local dev stack
set -euo pipefail
cd "$(dirname "$0")"
echo "[start] Starting VAIA L64 stack (Docker Compose)..."
docker compose up --build -d
echo "[start] Waiting for agent readiness..."
for i in $(seq 1 24); do
  if curl -sf http://localhost:8000/health/ready > /dev/null 2>&1; then
    echo "[start] ✓ Agent is ready!"
    break
  fi
  echo "[start] Waiting... ($i/24)"
  sleep 5
done
echo ""
echo "  Dashboard:   http://localhost:3000"
echo "  API docs:    http://localhost:8000/docs"
echo "  Metrics:     http://localhost:8000/metrics"
echo "  Prometheus:  http://localhost:9090"
