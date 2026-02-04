#!/bin/bash
# Cleanup: Stop containers, remove Docker resources, and clean project artifacts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "L32 ReAct Agent - Cleanup"
echo "=========================================="

# 1. Stop application services
echo ""
echo "1. Stopping application services..."
if [ -f "$SCRIPT_DIR/stop.sh" ]; then
    "$SCRIPT_DIR/stop.sh" 2>/dev/null || true
fi

# 2. Stop all Docker containers in this project
echo ""
echo "2. Stopping Docker containers..."
docker-compose down -v 2>/dev/null || true
docker compose down -v 2>/dev/null || true

# 3. Remove unused Docker resources
echo ""
echo "3. Removing unused Docker resources..."
docker container prune -f
docker image prune -af
docker network prune -f
docker volume prune -f
docker system prune -af --volumes 2>/dev/null || true

# 4. Remove project artifacts
echo ""
echo "4. Removing project artifacts..."
rm -rf frontend/node_modules
rm -rf backend/venv
rm -rf backend/.venv
rm -rf backend/.pytest_cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".eggs" -exec rm -rf {} + 2>/dev/null || true

# 5. Remove Istio-related files if any
echo ""
echo "5. Removing Istio files (if any)..."
find . -type f -name "*istio*" -delete 2>/dev/null || true
find . -type d -name "*istio*" -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "=========================================="
echo "✓ Cleanup complete"
echo "=========================================="
