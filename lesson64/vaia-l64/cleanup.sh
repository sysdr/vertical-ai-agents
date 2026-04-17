#!/usr/bin/env bash
set -euo pipefail

echo "[cleanup] Stopping local stack..."
docker compose down --remove-orphans || true

running_ids="$(docker ps -q)"
if [ -n "${running_ids}" ]; then
  echo "[cleanup] Stopping running containers..."
  docker stop ${running_ids} >/dev/null
fi

all_ids="$(docker ps -aq)"
if [ -n "${all_ids}" ]; then
  echo "[cleanup] Removing containers..."
  docker rm -f ${all_ids} >/dev/null || true
fi

echo "[cleanup] Pruning unused Docker resources..."
docker image prune -af >/dev/null || true
docker container prune -f >/dev/null || true
docker volume prune -f >/dev/null || true
docker network prune -f >/dev/null || true

echo "[cleanup] Done."
