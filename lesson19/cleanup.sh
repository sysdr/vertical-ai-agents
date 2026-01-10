#!/bin/bash
# Comprehensive Cleanup Script for Lesson 19 Project
# Stops containers and removes unused Docker resources, containers, and images
# Also removes node_modules, venv, cache files, and other build artifacts

set -e

echo "=================================================="
echo "Starting Comprehensive Cleanup Process"
echo "=================================================="

# Step 1: Stop and remove Docker Compose services
echo ""
echo "Step 1: Stopping Docker Compose services..."
if [ -f docker-compose.yml ]; then
    docker-compose down -v 2>/dev/null || true
    echo "✓ Docker Compose services stopped and removed"
fi

# Check for docker-compose files in subdirectories
for dir in */; do
    if [ -f "${dir}docker-compose.yml" ]; then
        echo "  Stopping services in ${dir}..."
        (cd "$dir" && docker-compose down -v 2>/dev/null || true)
    fi
done

# Step 2: Stop all running containers
echo ""
echo "Step 2: Stopping all running containers..."
RUNNING_CONTAINERS=$(docker ps -q 2>/dev/null || true)
if [ -n "$RUNNING_CONTAINERS" ]; then
    docker stop $RUNNING_CONTAINERS 2>/dev/null || true
    echo "✓ Stopped running containers"
else
    echo "✓ No running containers found"
fi

# Step 3: Remove all stopped containers
echo ""
echo "Step 3: Removing stopped containers..."
STOPPED_CONTAINERS=$(docker ps -aq 2>/dev/null || true)
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

# Step 7: Remove build cache
echo ""
echo "Step 7: Removing Docker build cache..."
docker builder prune -f 2>/dev/null || true
echo "✓ Cleaned up Docker build cache"

# Step 8: Clean up project-specific files
echo ""
echo "Step 8: Cleaning up project files..."

# Remove node_modules directories
echo "  Removing node_modules directories..."
# Try to change ownership first (if possible without sudo)
find . -type d -name "node_modules" -exec chown -R $USER:$USER {} + 2>/dev/null || true
find . -type d -name "node_modules" -exec chmod -R u+w {} + 2>/dev/null || true
find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true

# If still exists, try with sudo (may require password)
REMAINING_NM=$(find . -type d -name "node_modules" 2>/dev/null | wc -l)
if [ "$REMAINING_NM" -gt "0" ]; then
    echo "  ⚠ Some node_modules directories require sudo to remove (owned by root)"
    echo "    To remove manually, run: sudo rm -rf l19-naive-rag/frontend/node_modules"
else
    echo "  ✓ Removed node_modules"
fi

# Remove venv directories
echo "  Removing venv directories..."
find . -type d -name "venv" -exec chmod -R u+w {} + 2>/dev/null || true
find . -type d -name "venv" -exec rm -rf {} + 2>/dev/null || true
echo "  ✓ Removed venv"

# Remove __pycache__ directories
echo "  Removing Python cache files..."
find . -type d -name "__pycache__" -exec chmod -R u+w {} + 2>/dev/null || true
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo "  ✓ Removed Python cache files"

# Remove .pytest_cache
echo "  Removing .pytest_cache..."
find . -type d -name ".pytest_cache" -exec chmod -R u+w {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
echo "  ✓ Removed .pytest_cache"

# Remove .DS_Store files
echo "  Removing .DS_Store files..."
find . -type f -name ".DS_Store" -delete 2>/dev/null || true
echo "  ✓ Removed .DS_Store files"

# Remove Istio-related files
echo "  Removing Istio files..."
find . -type f \( -name "*istio*" -o -name "*Istio*" \) -delete 2>/dev/null || true
find . -type d -name "*istio*" -o -name "*Istio*" 2>/dev/null | while read dir; do
    [ -d "$dir" ] && rm -rf "$dir" 2>/dev/null || true
done
echo "  ✓ Removed Istio files (if any)"

# Remove PID files
echo "  Removing PID files..."
find . -type f \( -name "*.pid" -o -name ".backend.pid" -o -name ".frontend.pid" \) -delete 2>/dev/null || true
echo "  ✓ Removed PID files"

# Remove log files
echo "  Removing log files..."
find . -type f -name "*.log" -delete 2>/dev/null || true
echo "  ✓ Removed log files"

# Remove temporary files
echo "  Removing temporary files..."
find . -type f \( -name "*.tmp" -o -name "*.temp" -o -name "*.bak" -o -name "*.swp" -o -name "*.swo" \) -delete 2>/dev/null || true
echo "  ✓ Removed temporary files"

echo ""
echo "=================================================="
echo "✓ Cleanup Complete!"
echo "=================================================="
echo ""
echo "Summary:"
docker system df 2>/dev/null || echo "Docker system info unavailable"
echo ""
echo "Remaining directories check:"
REMAINING=$(find . -type d \( -name "node_modules" -o -name "venv" -o -name "__pycache__" \) 2>/dev/null | wc -l)
if [ "$REMAINING" -gt "0" ]; then
    echo "⚠ Some directories may still exist due to permission issues."
    echo "   If needed, manually remove with: sudo rm -rf <directory>"
else
    echo "✓ All target directories removed successfully"
fi
echo ""
echo "To start services again, run: ./start.sh (if available)"

