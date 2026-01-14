#!/bin/bash

# L21 Cleanup Script
# Stops containers and removes unused Docker resources

set -e

echo "🧹 Starting cleanup process..."

# Stop all running containers
echo "Stopping Docker containers..."
docker ps -q | xargs -r docker stop 2>/dev/null || true

# Stop docker-compose services
echo "Stopping docker-compose services..."
docker-compose down 2>/dev/null || true

# Remove stopped containers
echo "Removing stopped containers..."
docker container prune -f

# Remove unused images
echo "Removing unused Docker images..."
docker image prune -a -f

# Remove unused volumes
echo "Removing unused Docker volumes..."
docker volume prune -f

# Remove unused networks
echo "Removing unused Docker networks..."
docker network prune -f

# Remove build cache
echo "Removing Docker build cache..."
docker builder prune -a -f

echo "✅ Docker cleanup complete!"

