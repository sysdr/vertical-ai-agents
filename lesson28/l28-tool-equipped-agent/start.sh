#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env so GEMINI_API_KEY is available to backend
if [ -f "$SCRIPT_DIR/backend/.env" ]; then
    set -a
    source "$SCRIPT_DIR/backend/.env"
    set +a
fi

cd "$SCRIPT_DIR/backend"
source venv/bin/activate

echo "Starting L28 Tool-Equipped Agent..."

# Warn if API key looks like placeholder
KEY="${GEMINI_API_KEY:-}"
if [ -z "$KEY" ] || [ "$KEY" = "your_gemini_api_key_here" ]; then
    echo "⚠️  WARNING: Set GEMINI_API_KEY in backend/.env"
    echo "   Get key: https://aistudio.google.com/apikey"
fi

# Start backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend
sleep 3

# Start frontend
cd "$SCRIPT_DIR/frontend"
BROWSER=none npm start &
FRONTEND_PID=$!

echo "Application started!"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "CLI: python backend/cli.py"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
