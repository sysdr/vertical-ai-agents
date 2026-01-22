#!/bin/bash
set -e

echo "=== L24 LangChain RAG Cleanup Script ==="

# Stop and remove containers
echo "Stopping and removing containers..."
cd "$(dirname "$0")"
docker compose down --remove-orphans 2>/dev/null || true

# Remove unused Docker resources
echo "Removing unused Docker resources..."
docker system prune -f

# Remove unused containers
echo "Removing stopped containers..."
docker container prune -f

# Remove unused images (except those in use)
echo "Removing unused images..."
docker image prune -f

# Remove unused volumes
echo "Removing unused volumes..."
docker volume prune -f

# Remove unused networks
echo "Removing unused networks..."
docker network prune -f

# Remove project-specific images
echo "Removing project-specific images..."
docker images | grep "l24-langchain-rag" | awk '{print $3}' | xargs -r docker rmi -f 2>/dev/null || true

# Remove node_modules, venv, cache files
echo "Removing node_modules, venv, and cache files..."
# Remove node_modules (try multiple methods)
if [ -d "./frontend/node_modules" ]; then
    # Try to change ownership first if possible
    chown -R $USER:$USER ./frontend/node_modules 2>/dev/null || true
    rm -rf ./frontend/node_modules 2>/dev/null || echo "  Note: Some node_modules files may require sudo to remove"
fi
find . -type d -name "node_modules" ! -path "./frontend/node_modules" -prune -exec rm -rf {} + 2>/dev/null || true
# Remove Python virtual environments
find . -type d -name "venv" -prune -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".venv" -prune -exec rm -rf {} + 2>/dev/null || true
# Remove Python cache
find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true
# Remove pytest cache
find . -type d -name ".pytest_cache" -prune -exec rm -rf {} + 2>/dev/null || true
# Remove Istio files
find . -name "*istio*" -type f -delete 2>/dev/null || true
find . -type d -name "istio" -prune -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "=== Cleanup Complete ==="
echo "Removed:"
echo "  - Stopped containers"
echo "  - Unused Docker resources"
echo "  - Unused images, volumes, and networks"
echo "  - Project-specific Docker images"
echo "  - node_modules directories"
echo "  - Python venv directories"
echo "  - Python cache files (.pyc, __pycache__)"
echo "  - Pytest cache directories"
echo "  - Istio files"

