#!/bin/bash
# Run from project root (directory containing backend/, frontend/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting L39 Validator Compliance System..."

# Check for duplicate services on ports 8000 and 3000
if command -v lsof &> /dev/null; then
  for port in 8000 3000; do
    if lsof -i :$port -sTCP:LISTEN -t &>/dev/null; then
      echo "⚠️  Port $port already in use. Running stop.sh first..."
      [ -f "$SCRIPT_DIR/stop.sh" ] && "$SCRIPT_DIR/stop.sh" || true
      sleep 2
      break
    fi
  done
fi

# Check if Docker is available
if command -v docker &> /dev/null && [ -f "$SCRIPT_DIR/docker-compose.yml" ]; then
    echo "🐳 Starting with Docker..."
    docker-compose up -d
    
    echo "⏳ Waiting for services to be ready..."
    sleep 5
    
    echo "✅ System started!"
    echo "📊 Dashboard: http://localhost:3000"
    echo "🔧 API: http://localhost:8000"
    echo "📖 API Docs: http://localhost:8000/docs"
    echo ""
    echo "To view logs: docker-compose logs -f"
else
    echo "🐍 Starting without Docker..."
    
    # Start Redis if available
    if command -v redis-server &> /dev/null; then
        redis-server --daemonize yes 2>/dev/null || true
        echo "✅ Redis started (or already running)"
    else
        echo "⚠️  Redis not found - install with: brew install redis (Mac) or apt-get install redis-server (Linux)"
    fi
    
    # Start backend from project root so rules path resolves
    ( cd "$SCRIPT_DIR/backend" && source venv/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 8000 ) &
    BACKEND_PID=$!
    
    # Start frontend
    ( cd "$SCRIPT_DIR/frontend" && BROWSER=none npm start ) &
    FRONTEND_PID=$!
    
    echo "✅ System started!"
    echo "📊 Dashboard: http://localhost:3000"
    echo "🔧 API: http://localhost:8000"
    echo ""
    echo "To stop: $SCRIPT_DIR/stop.sh or ./stop.sh"
    
    # Save PIDs
    echo $BACKEND_PID > "$SCRIPT_DIR/.backend.pid"
    echo $FRONTEND_PID > "$SCRIPT_DIR/.frontend.pid"
fi
