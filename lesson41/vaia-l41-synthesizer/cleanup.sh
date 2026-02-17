#!/bin/bash
# Cleanup: stop services, Docker, and remove build/cache artifacts.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Stopping application services ==="
[ -f ./stop.sh ] && ./stop.sh || true
pkill -f "python.*main.py" 2>/dev/null || true
pkill -f "react-scripts" 2>/dev/null || true
pkill -f "node.*react-scripts" 2>/dev/null || true
echo "Done."

echo ""
echo "=== Stopping Docker containers and cleaning Docker resources ==="
docker compose -f ./docker-compose.yml down --remove-orphans 2>/dev/null || true
docker stop $(docker ps -q) 2>/dev/null || true
docker rm -f $(docker ps -aq) 2>/dev/null || true
docker system prune -af --volumes 2>/dev/null || true
echo "Docker cleanup done."

echo ""
echo "=== Removing node_modules, venv, .pytest_cache, .pyc, __pycache__ ==="
find "$SCRIPT_DIR" -type d -name "node_modules" -print0 2>/dev/null | xargs -0 -r rm -rf
find "$SCRIPT_DIR" -type d -name "venv" -print0 2>/dev/null | xargs -0 -r rm -rf
find "$SCRIPT_DIR" -type d -name ".pytest_cache" -print0 2>/dev/null | xargs -0 -r rm -rf
find "$SCRIPT_DIR" -type d -name "__pycache__" -print0 2>/dev/null | xargs -0 -r rm -rf
find "$SCRIPT_DIR" -name "*.pyc" -delete 2>/dev/null || true
# Istio-generated or addon files under this project
find "$SCRIPT_DIR" -maxdepth 3 -type f \( -name "*istio*.yaml" -o -name "*istio*.yml" \) 2>/dev/null | xargs -r rm -f
echo "Artifacts removed."

echo ""
echo "=== Cleanup complete ==="
