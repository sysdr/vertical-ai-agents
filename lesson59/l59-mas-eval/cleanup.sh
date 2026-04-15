#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$ROOT_DIR"

echo "==> Cleanup: stopping local services"
if [[ -x "$PROJECT_DIR/scripts/stop.sh" ]]; then
  bash "$PROJECT_DIR/scripts/stop.sh" || true
fi

echo "==> Cleanup: stopping Docker Compose stack (if present)"
if command -v docker >/dev/null 2>&1; then
  if [[ -f "$PROJECT_DIR/docker-compose.yml" ]]; then
    docker compose -f "$PROJECT_DIR/docker-compose.yml" down --remove-orphans || true
  fi

  echo "==> Cleanup: removing stopped containers"
  docker container prune -f || true

  echo "==> Cleanup: removing unused images"
  docker image prune -af || true

  echo "==> Cleanup: removing unused Docker resources"
  docker system prune -af --volumes || true
else
  echo "[WARN] Docker not installed; skipping Docker cleanup"
fi

echo "==> Cleanup complete"
