#!/bin/bash
echo "Starting L30 RAG Observability..."

# Ensure we run from project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"

# Prefer Docker when available (avoids venv issues)
if command -v docker-compose &> /dev/null && docker info &> /dev/null; then
    echo "Starting with Docker..."
    docker-compose up -d
    echo ""
    echo "✅ Services started!"
    echo "📊 Dashboard: http://localhost:3000"
    echo "🔌 API: http://localhost:8000"
    echo "📖 API Docs: http://localhost:8000/docs"
else
    echo "Starting without Docker..."
    
    # Start backend (from project root, PYTHONPATH=backend for imports, cwd for data/)
    cd "$PROJECT_ROOT"
    python -m venv backend/venv 2>/dev/null || true
    source backend/venv/bin/activate 2>/dev/null || source backend/venv/Scripts/activate
    pip install -q -r backend/requirements.txt
    PYTHONPATH="$PROJECT_ROOT/backend" uvicorn app.main:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    cd "$PROJECT_ROOT"
    
    # Start frontend
    cd "$PROJECT_ROOT/frontend"
    npm install
    npm start &
    FRONTEND_PID=$!
    cd "$PROJECT_ROOT"
    
    echo ""
    echo "✅ Services started!"
    echo "Backend PID: $BACKEND_PID"
    echo "Frontend PID: $FRONTEND_PID"
    echo "📊 Dashboard: http://localhost:3000"
    echo "🔌 API: http://localhost:8000"
fi
