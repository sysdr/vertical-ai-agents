#!/bin/bash

# Docker Cleanup Script
# Stops all containers and removes unused Docker resources

set -e

echo "🧹 Starting Docker cleanup..."

# Stop all running containers
echo "🛑 Stopping all running containers..."
if [ $(docker ps -q | wc -l) -gt 0 ]; then
    docker stop $(docker ps -q)
    echo "✅ All containers stopped"
else
    echo "ℹ️  No running containers found"
fi

# Remove all stopped containers
echo "🗑️  Removing stopped containers..."
if [ $(docker ps -aq | wc -l) -gt 0 ]; then
    docker rm $(docker ps -aq) 2>/dev/null || echo "⚠️  Some containers could not be removed (may be in use)"
    echo "✅ Stopped containers removed"
else
    echo "ℹ️  No stopped containers found"
fi

# Remove unused images
echo "🖼️  Removing unused images..."
docker image prune -a -f
echo "✅ Unused images removed"

# Remove unused volumes
echo "💾 Removing unused volumes..."
docker volume prune -f
echo "✅ Unused volumes removed"

# Remove unused networks
echo "🌐 Removing unused networks..."
docker network prune -f
echo "✅ Unused networks removed"

# Remove build cache
echo "📦 Removing build cache..."
docker builder prune -f
echo "✅ Build cache removed"

# Show summary
echo ""
echo "📊 Docker system summary:"
docker system df

echo ""
echo "✨ Docker cleanup completed!"


