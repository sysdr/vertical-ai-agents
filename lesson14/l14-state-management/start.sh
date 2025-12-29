#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting L14 State Management System..."

# Check if services are already running
if pgrep -f "uvicorn app.main:app" > /dev/null; then
    echo "⚠️  Backend service already running. Stopping existing instance..."
    pkill -f "uvicorn app.main:app" || true
    sleep 2
fi

if pgrep -f "react-scripts start" > /dev/null || pgrep -f "npm start" > /dev/null; then
    echo "⚠️  Frontend service already running. Stopping existing instance..."
    pkill -f "react-scripts start" || true
    pkill -f "npm start" || true
    sleep 2
fi

# Stop existing Docker containers if they exist
docker stop l14-postgres l14-redis 2>/dev/null || true
docker rm l14-postgres l14-redis 2>/dev/null || true

# Start PostgreSQL (requires Docker)
echo "Starting PostgreSQL..."
docker run -d --name l14-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=l14_state_db \
  -p 5432:5432 \
  postgres:16 || true

# Start Redis (requires Docker)
echo "Starting Redis..."
docker run -d --name l14-redis \
  -p 6379:6379 \
  redis:7-alpine || true

# Wait for databases
echo "Waiting for databases to be ready..."
sleep 5

# Check if venv exists
if [ ! -d "$SCRIPT_DIR/backend/venv" ]; then
    echo "❌ Virtual environment not found. Please run ./build.sh first"
    exit 1
fi

# Start backend
echo "Starting backend..."
cd "$SCRIPT_DIR/backend"
source venv/bin/activate
uvicorn app.main:app --reload --port 8000 --host 0.0.0.0 &
BACKEND_PID=$!
cd "$SCRIPT_DIR"

# Wait a bit for backend to start
sleep 3

# Start frontend
echo "Starting frontend..."
cd "$SCRIPT_DIR/frontend"
npm start &
FRONTEND_PID=$!
cd "$SCRIPT_DIR"

echo "✅ System started!"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "Docs: http://localhost:8000/docs"
echo ""
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Press Ctrl+C to stop all services"

wait $BACKEND_PID $FRONTEND_PID
