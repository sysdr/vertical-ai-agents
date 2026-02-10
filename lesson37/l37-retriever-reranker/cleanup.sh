#!/bin/bash
# cleanup.sh - Stop containers and remove unused Docker resources (run from l37-retriever-reranker)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Docker cleanup (l37-retriever-reranker) ==="

# Stop compose project if present
if [ -f "docker-compose.yml" ]; then
  echo "Stopping docker-compose..."
  docker compose down --remove-orphans 2>/dev/null || true
fi

# Stop all running containers
echo "Stopping all running containers..."
docker ps -q | xargs -r docker stop 2>/dev/null || true

# Remove all stopped containers
echo "Removing stopped containers..."
docker container prune -f

# Remove unused images (dangling and optionally unused)
echo "Removing dangling images..."
docker image prune -f
echo "Removing unused images (not used by any container)..."
docker image prune -a -f

# Remove unused volumes
echo "Removing unused volumes..."
docker volume prune -f

# Remove unused networks
echo "Removing unused networks..."
docker network prune -f

# Optional: full system prune (uncomment to use)
# docker system prune -a -f --volumes

echo "=== Cleanup complete ==="
