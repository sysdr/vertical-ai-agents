#!/bin/bash
set -e

echo "🚀 Starting L11 CoT Reasoning Evaluator..."

if command -v docker &> /dev/null && docker ps &> /dev/null; then
    echo "Starting with Docker..."
    if command -v docker-compose &> /dev/null; then
        docker-compose up -d
    elif docker compose version &> /dev/null; then
        docker compose up -d
    else
        echo "Docker Compose not available, starting without Docker..."
        START_WITHOUT_DOCKER=true
    fi
else
    START_WITHOUT_DOCKER=true
fi

if [ "$START_WITHOUT_DOCKER" = "true" ]; then
    echo "Starting without Docker..."
    
    # Check for GEMINI_API_KEY
    if [ -z "$GEMINI_API_KEY" ]; then
        echo "⚠️  WARNING: GEMINI_API_KEY environment variable is not set!"
        echo ""
        echo "Please set your Gemini API key:"
        echo "  export GEMINI_API_KEY='your-api-key-here'"
        echo ""
        echo "Or create a .env file in the cot-reasoning-evaluator directory with:"
        echo "  GEMINI_API_KEY=your-api-key-here"
        echo ""
        echo "You can get an API key from: https://makersuite.google.com/app/apikey"
        echo ""
        read -p "Press Enter to continue anyway (will fail if API key is not set) or Ctrl+C to cancel..."
    fi
    
    # Get absolute path
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # Load .env file if it exists
    if [ -f "$SCRIPT_DIR/.env" ]; then
        echo "Loading environment variables from .env file..."
        export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
    fi
    
    # Check if services are already running
    if lsof -i :8000 >/dev/null 2>&1; then
        echo "⚠ Backend already running on port 8000"
    else
        # Start backend
        cd "$SCRIPT_DIR/backend"
        if [ -d "venv" ]; then
            source venv/bin/activate
        fi
        python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &
        BACKEND_PID=$!
        cd ..
        echo "✓ Backend started (PID: $BACKEND_PID)"
    fi
    
    if lsof -i :3000 >/dev/null 2>&1; then
        echo "⚠ Frontend already running on port 3000"
    else
        # Start frontend
        cd "$SCRIPT_DIR/frontend"
        BROWSER=none npm start > /tmp/frontend.log 2>&1 &
        FRONTEND_PID=$!
        cd ..
        echo "✓ Frontend started (PID: $FRONTEND_PID)"
    fi
    
    echo ""
    echo "✅ Services running:"
    echo "   Backend API: http://localhost:8000"
    echo "   Frontend UI: http://localhost:3000"
    echo ""
    echo "View backend logs: tail -f /tmp/backend.log"
    echo "View frontend logs: tail -f /tmp/frontend.log"
    echo "Stop services: ./stop.sh"
fi
