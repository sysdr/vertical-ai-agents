#!/bin/bash

# L34 Cleanup Script - Stop containers and remove unused Docker resources

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🧹 L34 Cleanup - Stopping services and cleaning Docker..."

# Stop Python/Node services
echo "Stopping Python backend..."
pkill -f "python3 main.py" 2>/dev/null || true
pkill -f "python main.py" 2>/dev/null || true

echo "Stopping React frontend..."
pkill -f "react-scripts start" 2>/dev/null || true
pkill -f "node.*react-scripts" 2>/dev/null || true

# Stop Docker Compose
docker compose down 2>/dev/null || docker-compose down 2>/dev/null || true

# Stop all running containers
echo "Stopping Docker containers..."
docker stop $(docker ps -aq) 2>/dev/null || true

# Remove stopped containers
echo "Removing stopped containers..."
docker container prune -f

# Remove unused images
echo "Removing unused Docker images..."
docker image prune -a -f

# Remove unused volumes
echo "Removing unused volumes..."
docker volume prune -f

# Remove unused networks
echo "Removing unused networks..."
docker network prune -f

# Remove build cache
echo "Removing build cache..."
docker builder prune -f 2>/dev/null || true

echo ""
echo "✅ Cleanup complete!"
echo "   - Services stopped"
echo "   - Docker containers removed"
echo "   - Unused images, volumes, networks pruned"
