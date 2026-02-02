#!/bin/bash
# L29: Stop containers and remove unused Docker resources, node_modules, venv, caches

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "L29 Cleanup"
echo "========================================="

# Stop application services
echo "Stopping application services..."
pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true

# Docker: stop containers, remove unused resources
echo "Stopping Docker containers..."
docker compose down 2>/dev/null || true
docker stop $(docker ps -q) 2>/dev/null || true

echo "Removing unused Docker resources..."
docker container prune -f
docker image prune -af
docker volume prune -f
docker network prune -f
docker system prune -af --volumes 2>/dev/null || true

# Remove node_modules, venv, caches
echo "Removing node_modules, venv, caches..."
rm -rf frontend/node_modules
rm -rf backend/venv
rm -rf backend/.pytest_cache
find . -type d -name "__pycache__" -print0 2>/dev/null | xargs -0 rm -rf 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Remove .env to avoid committing API keys (use .env.example as template)
rm -f backend/.env 2>/dev/null || true

echo "✓ Cleanup complete"
