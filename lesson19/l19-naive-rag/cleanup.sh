#!/bin/bash
# Cleanup script for Naïve RAG project
# Stops containers and removes unused Docker resources, containers, and images

set -e

echo "=================================================="
echo "Starting Cleanup Process"
echo "=================================================="

# Step 1: Stop and remove Docker Compose services
echo ""
echo "Step 1: Stopping Docker Compose services..."
if [ -f docker-compose.yml ]; then
    docker-compose down -v 2>/dev/null || true
    echo "✓ Docker Compose services stopped and removed"
else
    echo "⚠ docker-compose.yml not found, skipping..."
fi

# Step 2: Stop all running containers
echo ""
echo "Step 2: Stopping all running containers..."
RUNNING_CONTAINERS=$(docker ps -q)
if [ -n "$RUNNING_CONTAINERS" ]; then
    docker stop $RUNNING_CONTAINERS 2>/dev/null || true
    echo "✓ Stopped running containers"
else
    echo "✓ No running containers found"
fi

# Step 3: Remove all stopped containers
echo ""
echo "Step 3: Removing stopped containers..."
STOPPED_CONTAINERS=$(docker ps -aq)
if [ -n "$STOPPED_CONTAINERS" ]; then
    docker rm $STOPPED_CONTAINERS 2>/dev/null || true
    echo "✓ Removed stopped containers"
else
    echo "✓ No stopped containers found"
fi

# Step 4: Remove unused Docker images
echo ""
echo "Step 4: Removing unused Docker images..."
docker image prune -a -f 2>/dev/null || true
echo "✓ Cleaned up unused Docker images"

# Step 5: Remove unused Docker volumes
echo ""
echo "Step 5: Removing unused Docker volumes..."
docker volume prune -f 2>/dev/null || true
echo "✓ Cleaned up unused Docker volumes"

# Step 6: Remove unused Docker networks
echo ""
echo "Step 6: Removing unused Docker networks..."
docker network prune -f 2>/dev/null || true
echo "✓ Cleaned up unused Docker networks"

# Step 7: Remove build cache (optional, can free significant space)
echo ""
echo "Step 7: Removing Docker build cache..."
docker builder prune -f 2>/dev/null || true
echo "✓ Cleaned up Docker build cache"

# Step 8: Clean up project-specific files
echo ""
echo "Step 8: Cleaning up project files..."

# Remove node_modules (with permission fix)
if [ -d "frontend/node_modules" ]; then
    echo "  Removing frontend/node_modules..."
    chmod -R u+w frontend/node_modules 2>/dev/null || true
    rm -rf frontend/node_modules 2>/dev/null || sudo rm -rf frontend/node_modules 2>/dev/null || true
    echo "  ✓ Removed node_modules"
fi

# Remove venv
if [ -d "venv" ]; then
    echo "  Removing venv..."
    chmod -R u+w venv 2>/dev/null || true
    rm -rf venv 2>/dev/null || sudo rm -rf venv 2>/dev/null || true
    echo "  ✓ Removed venv"
fi

# Remove __pycache__ directories
echo "  Removing Python cache files..."
find . -type d -name "__pycache__" -exec chmod -R u+w {} + 2>/dev/null || true
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo "  ✓ Removed Python cache files"

# Remove .pytest_cache
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
if [ $? -eq 0 ]; then
    echo "  ✓ Removed .pytest_cache"
fi

# Remove .DS_Store files
find . -type f -name ".DS_Store" -delete 2>/dev/null || true
echo "  ✓ Removed .DS_Store files"

# Remove Istio-related files (if any)
find . -type f -name "*istio*" -o -name "*Istio*" 2>/dev/null | while read file; do
    rm -f "$file" 2>/dev/null || true
done
echo "  ✓ Removed Istio files (if any)"

# Remove PID files
rm -f .backend.pid .frontend.pid 2>/dev/null || true
echo "  ✓ Removed PID files"

# Remove log files
rm -f backend.log *.log 2>/dev/null || true
echo "  ✓ Removed log files"

echo ""
echo "=================================================="
echo "✓ Cleanup Complete!"
echo "=================================================="
echo ""
echo "Summary:"
docker system df 2>/dev/null || echo "Docker system info unavailable"
echo ""

# Check for remaining directories that might need manual cleanup
REMAINING=$(find . -type d \( -name "node_modules" -o -name "venv" -o -name "__pycache__" \) 2>/dev/null | wc -l)
if [ "$REMAINING" -gt "0" ]; then
    echo "⚠ Note: Some directories (node_modules/venv) may still exist due to permission issues."
    echo "   If needed, manually remove with: sudo rm -rf frontend/node_modules venv"
    echo ""
fi

echo "To start services again, run: ./start.sh"

