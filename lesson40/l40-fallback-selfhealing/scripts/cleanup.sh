#!/bin/bash
# Cleanup: Stop containers, remove unused Docker resources, and project artifacts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== L40 Cleanup ==="

# Remove project artifacts
echo "Removing node_modules, venv, __pycache__, .pyc, .pytest_cache..."
rm -rf "$PROJECT_ROOT/frontend/node_modules"
rm -rf "$PROJECT_ROOT/backend/venv"
find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_ROOT" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_ROOT" -name "*.pyc" -delete 2>/dev/null || true

# Stop project containers
echo "Stopping project containers..."
cd "$PROJECT_ROOT"
docker-compose down -v 2>/dev/null || true

# Stop all running containers
echo "Stopping all Docker containers..."
docker stop $(docker ps -aq) 2>/dev/null || true

# Remove project containers
docker-compose rm -f 2>/dev/null || true

# Remove unused containers
echo "Removing stopped containers..."
docker container prune -f

# Remove unused images
echo "Removing unused images..."
docker image prune -a -f

# Remove unused volumes
echo "Removing unused volumes..."
docker volume prune -f

# Remove unused networks
echo "Removing unused networks..."
docker network prune -f

# Remove build cache (optional, saves disk)
echo "Removing build cache..."
docker builder prune -f 2>/dev/null || true

echo "=== Cleanup complete ==="
