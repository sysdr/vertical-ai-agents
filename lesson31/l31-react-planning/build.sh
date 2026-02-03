#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🏗️  Building L31 ReAct Planning System..."

# Docker build (primary path)
if command -v docker &> /dev/null; then
    echo "Building with Docker..."
    docker compose build 2>/dev/null || docker-compose build
    echo "✅ Docker build complete"
else
    echo "⚠️  Docker not found, skipping Docker build"
fi

# Local build (optional - for running without Docker)
if command -v docker &> /dev/null; then
    echo "Using Docker for runtime - skipping local venv/npm (optional)"
else
    echo "Setting up local environment..."
    # Backend
    if python3 -m venv "$SCRIPT_DIR/backend/venv" 2>/dev/null; then
        source "$SCRIPT_DIR/backend/venv/bin/activate"
        pip install -q -r "$SCRIPT_DIR/backend/requirements.txt"
        deactivate 2>/dev/null
        echo "  Backend venv OK"
    else
        echo "  ⚠️  Backend venv failed (install python3-venv: apt install python3.12-venv)"
    fi
    # Frontend
    (cd "$SCRIPT_DIR/frontend" && npm install 2>/dev/null) && echo "  Frontend npm OK" || echo "  ⚠️  Frontend npm failed (check permissions)"
fi

echo ""
echo "✅ Build complete!"
echo "Start with: ./start.sh"
