#!/bin/bash
# ============================================================
# L44 / Lesson44 — Cleanup script
# Stops containers, removes unused Docker resources,
# removes node_modules, venv, .pytest_cache, .pyc, __pycache__
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "🛑 Stopping L44 services..."
pkill -f "uvicorn app.main" 2>/dev/null && echo "   Backend stopped" || true
pkill -f "react-scripts start" 2>/dev/null && echo "   Frontend stopped" || true

echo ""
echo "🐳 Stopping Docker containers and removing unused resources..."
docker compose -f "$SCRIPT_DIR/docker-compose.yml" down 2>/dev/null || true
docker stop $(docker ps -q) 2>/dev/null || true
docker container prune -f 2>/dev/null || true
docker image prune -af 2>/dev/null || true
docker volume prune -f 2>/dev/null || true
docker network prune -f 2>/dev/null || true
docker system prune -af 2>/dev/null || true
echo "   Docker cleanup done."

echo ""
echo "🧹 Removing node_modules, venv, .pytest_cache, __pycache__, .pyc..."
rm -rf "$SCRIPT_DIR/frontend/node_modules" 2>/dev/null || true
rm -rf "$SCRIPT_DIR/venv" 2>/dev/null || true
rm -f "$SCRIPT_DIR/data/reports"/*.json 2>/dev/null || true
find "$SCRIPT_DIR" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find "$SCRIPT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$SCRIPT_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
# Remove Istio-related manifests if present (VirtualService, Gateway, DestinationRule, etc.)
find "$SCRIPT_DIR" -type f \( -name "*istio*" -o -name "*-vs.yaml" -o -name "*-gw.yaml" -o -name "*-dr.yaml" \) 2>/dev/null | while read -r f; do
  case "$f" in
    *node_modules*|*venv*|*/.git/*) ;;
    *) rm -f "$f"; echo "   Removed: $f" ;;
  esac
done 2>/dev/null || true
echo "   Cleanup done."

echo ""
echo "✅ Cleanup complete."
