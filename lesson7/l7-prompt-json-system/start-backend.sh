#!/bin/bash
# Start backend service

cd "$(dirname "$0")/backend"

# Check if venv exists
if [ ! -f "venv/bin/activate" ]; then
    echo "❌ Virtual environment not found. Run ./build.sh first."
    exit 1
fi

# Activate venv and start server
source venv/bin/activate

# Kill any existing backend processes
pkill -f "uvicorn main:app" 2>/dev/null
sleep 1

echo "🚀 Starting backend on http://localhost:8000"
uvicorn main:app --reload --host 0.0.0.0 --port 8000

