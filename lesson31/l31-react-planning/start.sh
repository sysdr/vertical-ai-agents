#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting L31 ReAct Planning System..."

# Check for duplicate services
if lsof -i :8000 -i :3000 2>/dev/null | grep -q LISTEN; then
    echo "⚠️  Ports 8000 or 3000 already in use. Stop existing services first: ./stop.sh"
    exit 1
fi

# Check for Docker
if command -v docker &> /dev/null && command -v docker compose &> /dev/null; then
    echo "Starting with Docker Compose..."
    docker compose up -d
    echo ""
    echo "✅ Services started!"
    echo "🌐 Frontend: http://localhost:3000"
    echo "🔧 Backend: http://localhost:8000"
    echo "📚 API Docs: http://localhost:8000/docs"
elif command -v docker-compose &> /dev/null; then
    echo "Starting with Docker Compose..."
    docker-compose up -d
    echo ""
    echo "✅ Services started!"
    echo "🌐 Frontend: http://localhost:3000"
    echo "🔧 Backend: http://localhost:8000"
    echo "📚 API Docs: http://localhost:8000/docs"
else
    echo "Starting locally..."
    
    # Start backend
    cd "$SCRIPT_DIR/backend"
    source venv/bin/activate || . venv/Scripts/activate
    uvicorn app.main:app --reload --port 8000 &
    BACKEND_PID=$!
    deactivate
    cd "$SCRIPT_DIR"
    
    # Start frontend
    cd "$SCRIPT_DIR/frontend"
    BROWSER=none npm start &
    FRONTEND_PID=$!
    cd "$SCRIPT_DIR"
    
    echo $BACKEND_PID > "$SCRIPT_DIR/.backend.pid"
    echo $FRONTEND_PID > "$SCRIPT_DIR/.frontend.pid"
    
    echo ""
    echo "✅ Services started!"
    echo "🌐 Frontend: http://localhost:3000"
    echo "🔧 Backend: http://localhost:8000"
    echo ""
    echo "Stop with: ./stop.sh"
fi
