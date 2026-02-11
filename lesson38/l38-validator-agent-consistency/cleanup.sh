#!/bin/bash
# Stop all app services, Docker containers; remove unused Docker resources
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo "=========================================="
echo "L38 ValidatorAgent - Cleanup"
echo "=========================================="

# 1. Stop app services (backend, frontend)
if [ -f "$SCRIPT_DIR/scripts/stop.sh" ]; then
  echo "Stopping app services..."
  "$SCRIPT_DIR/scripts/stop.sh" 2>/dev/null || true
fi

# 2. Stop Docker Compose (if used)
if [ -f "$PROJECT_ROOT/docker-compose.yml" ]; then
  echo "Stopping Docker Compose..."
  (cd "$PROJECT_ROOT" && docker compose down 2>/dev/null) || true
fi

# 3. Stop all running containers
echo "Stopping Docker containers..."
docker stop $(docker ps -q) 2>/dev/null || true

# 4. Remove stopped containers
echo "Removing stopped containers..."
docker container prune -f

# 5. Remove unused images (dangling)
echo "Removing dangling images..."
docker image prune -f

# 6. Remove unused volumes
echo "Removing unused volumes..."
docker volume prune -f

# 7. Remove unused networks
echo "Removing unused networks..."
docker network prune -f

# 8. Optional: full system prune (uncomment to remove all unused data)
# docker system prune -af --volumes

echo "=========================================="
echo "Cleanup complete."
echo "=========================================="
