#!/bin/bash
# L39 cleanup: stop containers, remove unused Docker resources, remove local build artifacts

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🧹 L39 Validator Compliance - Cleanup..."

# 1. Stop project containers
if [ -f "docker-compose.yml" ]; then
  echo "Stopping containers..."
  docker-compose down -v 2>/dev/null || true
fi

# 2. Stop any remaining running containers (for this project's ports)
echo "Stopping all Docker containers..."
docker stop $(docker ps -aq) 2>/dev/null || true

# 3. Remove unused Docker resources (containers, images, networks, build cache)
echo "Removing unused Docker resources..."
docker system prune -af --volumes 2>/dev/null || true

# 4. Remove node_modules
echo "Removing node_modules..."
find "$SCRIPT_DIR" -type d -name "node_modules" -print0 2>/dev/null | xargs -0 -r rm -rf 2>/dev/null || true

# 5. Remove venv
echo "Removing venv..."
find "$SCRIPT_DIR" -type d -name "venv" -print0 2>/dev/null | xargs -0 -r rm -rf 2>/dev/null || true

# 6. Remove .pytest_cache
echo "Removing .pytest_cache..."
find "$SCRIPT_DIR" -type d -name ".pytest_cache" -print0 2>/dev/null | xargs -0 -r rm -rf 2>/dev/null || true

# 7. Remove __pycache__ and .pyc files
echo "Removing __pycache__ and .pyc files..."
find "$SCRIPT_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$SCRIPT_DIR" -type d -name "__pycache__" -print0 2>/dev/null | xargs -0 -r rm -rf 2>/dev/null || true

# 8. Remove Istio-related files/dirs (if any)
echo "Removing Istio files..."
find "$SCRIPT_DIR" -type d -name "istio*" 2>/dev/null | while read -r d; do rm -rf "$d"; done 2>/dev/null || true
find "$SCRIPT_DIR" -type f -name "*istio*" -delete 2>/dev/null || true

# 9. Remove PID files
rm -f .backend.pid .frontend.pid 2>/dev/null || true

echo "✅ Cleanup complete."
