#!/bin/bash

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🚀 Starting L18 Tool Execution Loop..."
echo "📁 Project root: $PROJECT_ROOT"

# Check if backend venv exists
if [ ! -f "$PROJECT_ROOT/backend/venv/bin/activate" ]; then
    echo "❌ Backend virtual environment not found. Run ./scripts/build.sh first."
    exit 1
fi

# Check if frontend node_modules exists
if [ ! -d "$PROJECT_ROOT/frontend/node_modules" ]; then
    echo "❌ Frontend dependencies not found. Run ./scripts/build.sh first."
    exit 1
fi

# Check API key (optional warning)
if [ -f "$PROJECT_ROOT/backend/.env" ]; then
    echo "🔍 Validating API key..."
    cd "$PROJECT_ROOT/backend"
    source venv/bin/activate
    if ! python ../scripts/validate_api_key.py 2>/dev/null; then
        echo ""
        echo "⚠️  WARNING: API key validation failed!"
        echo "   The backend will start, but chat functionality won't work."
        echo "   To fix: Run ./scripts/setup_api_key.sh"
        echo ""
        read -p "Continue anyway? (y/N): " continue_anyway
        if [[ ! $continue_anyway =~ ^[Yy]$ ]]; then
            echo "Cancelled. Fix API key first, then try again."
            exit 1
        fi
    else
        echo "✅ API key is valid"
    fi
    cd "$PROJECT_ROOT"
else
    echo "⚠️  WARNING: No API key configured!"
    echo "   The backend will start, but chat functionality won't work."
    echo "   To fix: Run ./scripts/setup_api_key.sh"
    echo ""
fi

# Start backend
cd "$PROJECT_ROOT/backend"
source venv/bin/activate
python main.py &
BACKEND_PID=$!
cd "$PROJECT_ROOT"

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 5

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start"
    exit 1
fi

# Start frontend
cd "$PROJECT_ROOT/frontend"
npm run dev &
FRONTEND_PID=$!
cd "$PROJECT_ROOT"

# Wait a bit for frontend
sleep 2

echo "✅ Services running:"
echo "   Backend PID: $BACKEND_PID"
echo "   Frontend PID: $FRONTEND_PID"
echo "   Backend: http://localhost:8000"
echo "   Frontend: http://localhost:3000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for Ctrl+C
trap "echo '🛑 Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
