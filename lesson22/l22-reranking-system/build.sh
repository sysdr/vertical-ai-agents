#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔨 Building L22 Reranking System..."
echo "📁 Working directory: $SCRIPT_DIR"

# Check Docker
if command -v docker &> /dev/null && [ -f "$SCRIPT_DIR/docker-compose.yml" ]; then
    echo "📦 Building with Docker..."
    docker-compose -f "$SCRIPT_DIR/docker-compose.yml" build
    echo "✅ Docker build complete"
else
    echo "📦 Building without Docker..."
    
    # Backend
    BACKEND_DIR="$SCRIPT_DIR/backend"
    if [ -f "$BACKEND_DIR/requirements.txt" ]; then
        cd "$BACKEND_DIR"
        if [ ! -d "$BACKEND_DIR/venv" ]; then
            python3 -m venv venv
        fi
        source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null || true
        pip install -q -r requirements.txt
        echo "✅ Backend dependencies installed"
        cd "$SCRIPT_DIR"
    else
        echo "⚠️  Backend requirements.txt not found"
    fi
    
    # Frontend
    FRONTEND_DIR="$SCRIPT_DIR/frontend"
    if [ -f "$FRONTEND_DIR/package.json" ]; then
        cd "$FRONTEND_DIR"
        npm install
        echo "✅ Frontend dependencies installed"
        cd "$SCRIPT_DIR"
    else
        echo "⚠️  Frontend package.json not found"
    fi
    
    echo "✅ Build complete"
fi
