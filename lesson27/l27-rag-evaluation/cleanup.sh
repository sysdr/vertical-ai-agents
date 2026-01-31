#!/bin/bash
##############################################################################
# L27 Cleanup Script - Stop containers, remove Docker resources
##############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo "========================================="
echo "L27 Cleanup - Docker & Project Resources"
echo "========================================="

# Stop L27 services
echo ""
echo "Stopping L27 services..."
pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
echo "  Done."

# Stop all Docker containers
echo ""
echo "Stopping Docker containers..."
docker stop $(docker ps -aq) 2>/dev/null || true
docker compose down 2>/dev/null || true
echo "  Done."

# Remove Docker containers
echo ""
echo "Removing Docker containers..."
docker rm -f $(docker ps -aq) 2>/dev/null || true
echo "  Done."

# Remove unused Docker images (dangling)
echo ""
echo "Removing dangling Docker images..."
docker image prune -f
echo "  Done."

# Remove unused Docker images (all unused, not just dangling)
echo ""
echo "Removing unused Docker images..."
docker image prune -a -f
echo "  Done."

# Remove unused Docker volumes
echo ""
echo "Removing unused Docker volumes..."
docker volume prune -f
echo "  Done."

# Remove unused Docker networks
echo ""
echo "Removing unused Docker networks..."
docker network prune -f
echo "  Done."

# Remove build cache
echo ""
echo "Removing Docker build cache..."
docker builder prune -f 2>/dev/null || true
echo "  Done."

# Remove node_modules, venv, .pytest_cache, __pycache__, .pyc
echo ""
echo "Removing project artifacts..."
rm -rf "$PROJECT_ROOT/frontend/node_modules" 2>/dev/null || true
rm -rf "$PROJECT_ROOT/backend/venv" 2>/dev/null || true
rm -rf "$PROJECT_ROOT/backend/.pytest_cache" 2>/dev/null || true
find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_ROOT" -name "*.pyc" -delete 2>/dev/null || true
echo "  Done."

echo ""
echo "========================================="
echo "✅ Cleanup complete!"
echo "========================================="
