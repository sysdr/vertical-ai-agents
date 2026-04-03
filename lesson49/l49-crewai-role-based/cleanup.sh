#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Lesson49 (l49-crewai-role-based) Cleanup ==="

echo "Stopping compose services (if present)..."
if [ -f "${ROOT_DIR}/docker-compose.yml" ]; then
  docker compose -f "${ROOT_DIR}/docker-compose.yml" down --remove-orphans || true
fi

echo "Stopping all running docker containers..."
if command -v docker >/dev/null 2>&1; then
  docker ps -q | xargs -r docker stop
fi

echo "Removing local build/cache artifacts..."
find "${ROOT_DIR}" -type d -name "node_modules" -prune -exec rm -rf {} + 2>/dev/null || true
find "${ROOT_DIR}" -type d -name ".venv" -prune -exec rm -rf {} + 2>/dev/null || true
find "${ROOT_DIR}" -type d -name ".pytest_cache" -prune -exec rm -rf {} + 2>/dev/null || true
find "${ROOT_DIR}" -type f -name "*.pyc" -delete 2>/dev/null || true
find "${ROOT_DIR}" -type f -iname "*istio*" -delete 2>/dev/null || true
find "${ROOT_DIR}" -type d -iname "*istio*" -prune -exec rm -rf {} + 2>/dev/null || true

echo "Pruning unused Docker resources..."
docker container prune -f || true
docker image prune -af || true
docker network prune -f || true
docker volume prune -f || true

echo "Cleanup complete."
