#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$ROOT/l54-agent-personalization"
COMPOSE_FILE="$APP_DIR/docker/docker-compose.yml"

echo "[cleanup] Stopping local app services..."
if [ -f "$APP_DIR/scripts/stop.sh" ]; then
  bash "$APP_DIR/scripts/stop.sh" || true
fi

echo "[cleanup] Stopping docker compose stack..."
if [ -f "$COMPOSE_FILE" ]; then
  docker compose -f "$COMPOSE_FILE" down --remove-orphans || true
fi

echo "[cleanup] Removing stopped containers..."
docker container prune -f || true

echo "[cleanup] Removing unused images..."
docker image prune -af || true

echo "[cleanup] Removing unused networks/volumes/build cache..."
docker system prune -af --volumes || true

echo "[cleanup] Done."
