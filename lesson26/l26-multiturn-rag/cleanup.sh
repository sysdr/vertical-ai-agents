#!/bin/bash

###############################################################################
# L26: Multi-Turn RAG - Cleanup Script
# Stops containers and removes unused Docker resources
###############################################################################

set -e

echo "================================================"
echo "L26: Docker Cleanup"
echo "================================================"

# Stop all running containers
echo "Stopping all running containers..."
docker ps -q | xargs -r docker stop 2>/dev/null || true

# Remove all stopped containers
echo "Removing stopped containers..."
docker ps -a -q | xargs -r docker rm 2>/dev/null || true

# Remove unused images
echo "Removing unused images..."
docker image prune -a -f 2>/dev/null || true

# Remove unused volumes
echo "Removing unused volumes..."
docker volume prune -f 2>/dev/null || true

# Remove unused networks
echo "Removing unused networks..."
docker network prune -f 2>/dev/null || true

# Remove build cache
echo "Removing build cache..."
docker builder prune -f 2>/dev/null || true

# Stop and remove containers from docker-compose if docker-compose.yml exists
if [ -f docker-compose.yml ]; then
    echo "Stopping docker-compose services..."
    docker-compose down -v 2>/dev/null || true
fi

echo ""
echo "================================================"
echo "✅ Docker cleanup complete!"
echo "================================================"
echo ""
echo "Remaining Docker resources:"
docker ps -a
echo ""
docker images
echo ""
