#!/bin/bash
set -e

# Change to project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 Starting L34 Planning Loop Controls..."
echo "📁 Project root: $PROJECT_ROOT"

# Check if Docker is available
if command -v docker &> /dev/null && docker-compose version &> /dev/null; then
    echo "🐳 Starting with Docker Compose..."
    docker-compose up -d
    
    echo ""
    echo "✅ Services started!"
    echo "🌐 Frontend: http://localhost:3000"
    echo "🔧 Backend API: http://localhost:8000"
    echo "📚 API Docs: http://localhost:8000/docs"
    echo ""
    echo "📊 View logs: docker-compose logs -f"
    echo "🛑 Stop: docker-compose down"
else
    echo "🐍 Starting without Docker..."
    
    # Start backend
    cd backend
    if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        pip install -q -r requirements.txt
    else
        echo "⚠️  venv not found, using system Python"
        pip install -q -r requirements.txt 2>/dev/null || true
    fi
    
    export GEMINI_API_KEY="${GEMINI_API_KEY:-}"
    python3 main.py &
    BACKEND_PID=$!
    cd ..
    
    # Start frontend
    cd frontend
    [ ! -d "node_modules" ] && npm install
    npm start &
    FRONTEND_PID=$!
    cd ..
    
    echo ""
    echo "✅ Services started!"
    echo "🌐 Frontend: http://localhost:3000"
    echo "🔧 Backend: http://localhost:8000"
    echo ""
    echo "Backend PID: $BACKEND_PID"
    echo "Frontend PID: $FRONTEND_PID"
    echo ""
    echo "To stop: kill $BACKEND_PID $FRONTEND_PID"
fi
