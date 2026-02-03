#!/bin/bash
set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or not in PATH." >&2
  exit 1
fi

echo "Stopping running containers..."
if [ -n "$(docker ps -q)" ]; then
  docker stop $(docker ps -q)
else
  echo "No running containers to stop."
fi

echo "Removing unused containers, networks, and images..."
docker system prune -af

if docker volume ls -q >/dev/null 2>&1; then
  echo "Removing dangling volumes..."
  docker volume prune -f || true
fi

echo "Cleanup complete."
