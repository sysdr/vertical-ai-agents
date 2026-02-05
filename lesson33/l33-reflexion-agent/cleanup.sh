#!/bin/bash
# Cleanup script: Stop services, Docker, remove build artifacts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

echo "=== L33 Reflexion Agent Cleanup ==="

# 1. Stop application services
echo "Stopping application services..."
pkill -f "backend.api" 2>/dev/null || true
pkill -f "react-scripts" 2>/dev/null || true
[ -f "$PROJECT_ROOT/scripts/stop.sh" ] && bash "$PROJECT_ROOT/scripts/stop.sh" 2>/dev/null || true

# 2. Stop Docker containers and services
echo "Stopping Docker containers..."
docker stop $(docker ps -aq) 2>/dev/null || true

echo "Stopping Docker Compose (if any)..."
docker compose down 2>/dev/null || true
docker-compose down 2>/dev/null || true

# 3. Remove unused Docker resources
echo "Removing unused Docker resources..."
docker container prune -f 2>/dev/null || true
docker image prune -af 2>/dev/null || true
docker volume prune -f 2>/dev/null || true
docker network prune -f 2>/dev/null || true
docker system prune -af 2>/dev/null || true

# 4. Remove node_modules
echo "Removing node_modules..."
rm -rf "$PROJECT_ROOT/frontend/node_modules" 2>/dev/null || true

# 5. Remove venv
echo "Removing venv..."
rm -rf "$PROJECT_ROOT/venv" 2>/dev/null || true

# 6. Remove .pytest_cache
echo "Removing .pytest_cache..."
find "$PROJECT_ROOT" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

# 7. Remove .pyc files and __pycache__
echo "Removing .pyc and __pycache__..."
find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_ROOT" -type f -name "*.pyc" -delete 2>/dev/null || true

# 8. Remove Istio-related files/dirs (if any)
echo "Removing Istio files (if any)..."
find "$PROJECT_ROOT" -name "*istio*" 2>/dev/null | while read f; do rm -rf "$f" 2>/dev/null; done || true

echo ""
echo "✓ Cleanup complete"
