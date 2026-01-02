#!/bin/bash
# Cleanup script to stop containers and remove unused Docker resources

set -e

echo "=========================================="
echo "Docker Cleanup Script"
echo "=========================================="

# Stop all running containers
echo ""
echo "1. Stopping all running containers..."
if docker ps -q | wc -l | grep -qv "^0$"; then
    docker stop $(docker ps -q)
    echo "   ✓ Stopped running containers"
else
    echo "   ✓ No running containers"
fi

# Remove all stopped containers
echo ""
echo "2. Removing stopped containers..."
if docker ps -aq | wc -l | grep -qv "^0$"; then
    docker rm $(docker ps -aq) 2>/dev/null || echo "   ⚠️  Some containers couldn't be removed (may be in use)"
    echo "   ✓ Removed stopped containers"
else
    echo "   ✓ No stopped containers"
fi

# Remove unused images
echo ""
echo "3. Removing unused Docker images..."
docker image prune -af --filter "until=24h" 2>/dev/null || echo "   ⚠️  No unused images to remove"
echo "   ✓ Cleaned unused images"

# Remove unused volumes
echo ""
echo "4. Removing unused volumes..."
docker volume prune -af 2>/dev/null || echo "   ⚠️  No unused volumes to remove"
echo "   ✓ Cleaned unused volumes"

# Remove unused networks
echo ""
echo "5. Removing unused networks..."
docker network prune -af 2>/dev/null || echo "   ⚠️  No unused networks to remove"
echo "   ✓ Cleaned unused networks"

# System prune (optional - be careful with this)
echo ""
echo "6. System-wide cleanup (optional)..."
read -p "   Run 'docker system prune -a'? This removes ALL unused resources (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker system prune -af
    echo "   ✓ System-wide cleanup completed"
else
    echo "   ⏭️  Skipped system-wide cleanup"
fi

echo ""
echo "=========================================="
echo "✓ Docker cleanup complete!"
echo "=========================================="

