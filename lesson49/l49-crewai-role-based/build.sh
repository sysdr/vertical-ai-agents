#!/bin/bash
set -e
echo "=== L49 Build ==="
if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
  echo "Building Docker images..."
  docker compose build
  echo "✅ Docker build complete"
else
  echo "Building without Docker..."
  cd backend && python -m venv .venv && source .venv/bin/activate && pip install -q -r requirements.txt && deactivate && cd ..
  cd frontend && npm install --silent && cd ..
  echo "✅ Local build complete"
fi
