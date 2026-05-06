#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$ROOT_DIR"

echo "=== Cleanup started ==="

echo "Stopping app services..."
if [ -x "$APP_DIR/stop.sh" ]; then
  bash "$APP_DIR/stop.sh" || true
fi

if command -v docker >/dev/null 2>&1; then
  echo "Stopping Docker Compose stack (if any)..."
  docker compose -f "$APP_DIR/docker-compose.yml" down --remove-orphans 2>/dev/null || true

  echo "Stopping running containers..."
  docker ps -q | xargs -r docker stop >/dev/null || true

  echo "Removing stopped containers..."
  docker container prune -f >/dev/null || true

  echo "Removing unused images, networks, and cache..."
  docker system prune -af >/dev/null || true
else
  echo "Docker not found; skipping Docker cleanup."
fi

echo "=== Cleanup complete ==="
