#!/bin/bash
# Cleanup script to stop containers and remove unused Docker resources

set -e

echo "🧹 Starting Docker cleanup..."

# Stop and remove containers
if [ -f "docker-compose.yml" ]; then
    echo "📦 Stopping Docker Compose services..."
    docker-compose down -v 2>/dev/null || true
fi

# Stop all running containers
echo "🛑 Stopping all running containers..."
docker stop $(docker ps -q) 2>/dev/null || echo "No running containers to stop"

# Remove stopped containers
echo "🗑️  Removing stopped containers..."
docker container prune -f

# Remove unused images
echo "🖼️  Removing unused Docker images..."
docker image prune -a -f

# Remove unused volumes
echo "💾 Removing unused Docker volumes..."
docker volume prune -f

# Remove unused networks
echo "🌐 Removing unused Docker networks..."
docker network prune -f

# Remove build cache
echo "🧼 Removing Docker build cache..."
docker builder prune -f

echo "✅ Docker cleanup complete!"

# Show remaining resources
echo ""
echo "📊 Remaining Docker resources:"
echo "Containers: $(docker ps -a -q | wc -l)"
echo "Images: $(docker images -q | wc -l)"
echo "Volumes: $(docker volume ls -q | wc -l)"
echo "Networks: $(docker network ls -q | wc -l)"

