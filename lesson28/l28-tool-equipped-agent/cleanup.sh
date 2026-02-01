#!/bin/bash
# Cleanup: stop services, Docker, and remove generated/cache files
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== L28 Cleanup ==="

# 1. Stop application services
if [ -f ./stop.sh ]; then
  echo "Stopping application services..."
  ./stop.sh 2>/dev/null || true
fi

# 2. Kill processes on common ports
for port in 8000 3000; do
  if command -v fuser &>/dev/null; then
    fuser -k ${port}/tcp 2>/dev/null || true
  fi
done

# 3. Stop all Docker containers
echo "Stopping Docker containers..."
docker stop $(docker ps -q) 2>/dev/null || true

# 4. Remove unused Docker resources
echo "Removing unused Docker resources..."
docker container prune -f 2>/dev/null || true
docker image prune -a -f 2>/dev/null || true
docker network prune -f 2>/dev/null || true
docker volume prune -f 2>/dev/null || true
docker system prune -a -f 2>/dev/null || true

# 5. Remove node_modules
echo "Removing node_modules..."
rm -rf frontend/node_modules 2>/dev/null || true

# 6. Remove Python venv
echo "Removing venv..."
rm -rf backend/venv 2>/dev/null || true

# 7. Remove .pytest_cache
echo "Removing .pytest_cache..."
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

# 8. Remove __pycache__ and .pyc files
echo "Removing __pycache__ and .pyc files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# 9. Remove Istio-related config (project-level only)
echo "Removing Istio files..."
find . -maxdepth 4 -type d -name "istio" ! -path "*/node_modules/*" 2>/dev/null | while read d; do rm -rf "$d"; done

echo "=== Cleanup complete ==="
