#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting L22 Reranking System..."
echo "📁 Working directory: $SCRIPT_DIR"

if command -v docker &> /dev/null && [ -f "$SCRIPT_DIR/docker-compose.yml" ]; then
    echo "🐳 Starting with Docker..."
    docker-compose -f "$SCRIPT_DIR/docker-compose.yml" up -d
    echo "✅ Services started"
    echo "📊 Dashboard: http://localhost:3000"
    echo "🔧 API: http://localhost:8000"
    echo "📖 Docs: http://localhost:8000/docs"
else
    echo "🔧 Starting without Docker..."
    
    # Start Redis (if installed)
    if command -v redis-server &> /dev/null; then
        redis-server --daemonize yes 2>/dev/null || echo "⚠️  Redis not available"
    else
        echo "⚠️  Redis not available"
    fi
    
    # Start backend
    BACKEND_DIR="$SCRIPT_DIR/backend"
    if [ -f "$BACKEND_DIR/main.py" ]; then
        cd "$BACKEND_DIR"
        if [ -d "$BACKEND_DIR/venv" ]; then
            source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null || true
        fi
        if [ -f "$BACKEND_DIR/requirements.txt" ] && [ ! -d "$BACKEND_DIR/venv" ]; then
            echo "📦 Creating virtual environment..."
            python3 -m venv venv
            source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
            pip install -q -r requirements.txt
        fi
        python main.py > "$SCRIPT_DIR/backend.log" 2>&1 &
        BACKEND_PID=$!
        echo "✅ Backend started (PID: $BACKEND_PID)"
        cd "$SCRIPT_DIR"
    else
        echo "❌ Backend main.py not found at $BACKEND_DIR/main.py"
        exit 1
    fi
    
    # Start frontend
    FRONTEND_DIR="$SCRIPT_DIR/frontend"
    if [ -f "$FRONTEND_DIR/package.json" ]; then
        cd "$FRONTEND_DIR"
        if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
            echo "📦 Installing frontend dependencies..."
            npm install
        fi
        npm start > "$SCRIPT_DIR/frontend.log" 2>&1 &
        FRONTEND_PID=$!
        echo "✅ Frontend started (PID: $FRONTEND_PID)"
        cd "$SCRIPT_DIR"
    else
        echo "❌ Frontend package.json not found at $FRONTEND_DIR/package.json"
        exit 1
    fi
    
    echo "✅ Services started"
    echo "Backend PID: $BACKEND_PID"
    echo "Frontend PID: $FRONTEND_PID"
    echo "📊 Dashboard: http://localhost:3000"
    echo "🔧 API: http://localhost:8000"
    echo "📝 Logs: backend.log, frontend.log"
fi
