#!/usr/bin/env bash
# Stops L69 local services, tears down Docker Compose for this app, and prunes
# unused Docker resources (see README.md in this directory).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "==> Stopping L69 shell-managed services (if any)"
if [[ -x stop.sh ]]; then
  bash stop.sh || true
else
  pkill -f "l69-agent-observability.*uvicorn main:app" 2>/dev/null || true
  pkill -f "l69-agent-observability/frontend/node_modules/react-scripts" 2>/dev/null || true
  pkill -f "uvicorn main:app.*8069" 2>/dev/null || true
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker not installed; skipping container cleanup."
  exit 0
fi

echo "==> Docker: compose down (this project)"
if [[ -f docker-compose.yml ]]; then
  docker compose down --remove-orphans 2>/dev/null \
    || docker-compose down --remove-orphans 2>/dev/null \
    || true
fi

echo "==> Docker: remove stopped containers, unused images, networks, build cache, dangling"
docker container prune -f
docker image prune -a -f
docker network prune -f
docker builder prune -f
docker system prune -f

echo "==> Done."
