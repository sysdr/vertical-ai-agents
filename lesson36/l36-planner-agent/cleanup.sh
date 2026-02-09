#!/bin/bash
# Cleanup script: Stop containers, remove unused Docker resources, and clean project artifacts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== L36 Cleanup ==="

# Stop L36 services
echo "Stopping L36 services..."
pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "react-scripts start" 2>/dev/null || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
echo "  Services stopped."

# Stop Docker containers
echo "Stopping Docker containers..."
docker stop $(docker ps -aq) 2>/dev/null || true
echo "  Containers stopped."

# Remove unused Docker resources
echo "Removing unused Docker resources..."
docker container prune -f 2>/dev/null || true
docker image prune -af 2>/dev/null || true
docker volume prune -f 2>/dev/null || true
docker network prune -f 2>/dev/null || true
docker system prune -af 2>/dev/null || true
echo "  Docker cleanup complete."

# Remove project artifacts
echo "Removing project artifacts..."
rm -rf frontend/node_modules 2>/dev/null || true
rm -rf backend/venv 2>/dev/null || true
rm -rf backend/.packages 2>/dev/null || true
rm -rf backend/.pytest_cache 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -path "*istio*" -type f -delete 2>/dev/null || true
find . -path "*istio*" -type d -empty -delete 2>/dev/null || true
echo "  Project artifacts removed."

echo ""
echo "Cleanup complete."
