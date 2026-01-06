#!/bin/bash

# Docker Cleanup Script
# Stops containers and removes unused Docker resources

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "Docker Cleanup Script"
echo "=================================================="
echo ""

# Stop all running containers
echo "Stopping all running containers..."
docker ps -q | xargs -r docker stop 2>/dev/null || echo "No running containers to stop"
echo "✓ Containers stopped"
echo ""

# Remove all stopped containers
echo "Removing stopped containers..."
docker ps -a -q | xargs -r docker rm 2>/dev/null || echo "No containers to remove"
echo "✓ Containers removed"
echo ""

# Remove unused images
echo "Removing unused images..."
docker image prune -a -f 2>/dev/null || echo "No unused images to remove"
echo "✓ Unused images removed"
echo ""

# Remove unused volumes
echo "Removing unused volumes..."
docker volume prune -f 2>/dev/null || echo "No unused volumes to remove"
echo "✓ Unused volumes removed"
echo ""

# Remove unused networks
echo "Removing unused networks..."
docker network prune -f 2>/dev/null || echo "No unused networks to remove"
echo "✓ Unused networks removed"
echo ""

# System prune (optional - removes everything unused)
read -p "Do you want to run 'docker system prune -a' (removes ALL unused resources)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running docker system prune..."
    docker system prune -a -f
    echo "✓ System prune complete"
else
    echo "Skipping system prune"
fi

echo ""
echo "=================================================="
echo "✓ Docker cleanup complete!"
echo "=================================================="
echo ""
echo "Docker status:"
docker ps -a
echo ""
docker images | head -5
echo ""

