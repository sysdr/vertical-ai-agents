#!/bin/bash
set -e

# Resolve script directory and project root for full-path execution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Starting L40 Fallback & Self-Healing..."
echo "Project root: $PROJECT_ROOT"

# Check for Docker
if command -v docker &> /dev/null; then
    echo "Starting with Docker..."
    cd "$PROJECT_ROOT"
    docker-compose up -d
    echo "✅ Services started!"
    echo "Backend: http://localhost:8000"
    echo "Frontend: http://localhost:3000"
else
    echo "Starting without Docker..."
    
    # Start Redis (if installed)
    if command -v redis-server &> /dev/null; then
        redis-server --daemonize yes 2>/dev/null || true
    fi
    
    # Start backend
    cd "$PROJECT_ROOT/backend"
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    pip install -q -r requirements.txt
    uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    cd "$PROJECT_ROOT"
    
    # Start frontend
    cd "$PROJECT_ROOT/frontend"
    npm install --silent 2>/dev/null || npm install
    BROWSER=none npm start &
    FRONTEND_PID=$!
    cd "$PROJECT_ROOT"
    
    echo "✅ Services started!"
    echo "Backend PID: $BACKEND_PID"
    echo "Frontend PID: $FRONTEND_PID"
    echo "Backend: http://localhost:8000"
    echo "Frontend: http://localhost:3000"
fi
