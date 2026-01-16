#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔍 Checking system status..."

# Check for duplicate services
echo "Checking for running services..."
RUNNING=$(docker ps --format "{{.Names}}" 2>/dev/null | grep -E "(backend|frontend|redis)" || echo "")
if [ -n "$RUNNING" ]; then
    echo "⚠️  Found running services:"
    echo "$RUNNING"
    echo "Stopping them..."
    docker-compose down 2>/dev/null || true
    sleep 2
fi

# Check if images exist
BACKEND_IMAGE=$(docker images --format "{{.Repository}}" 2>/dev/null | grep "l22-reranking-system-backend" || echo "")
FRONTEND_IMAGE=$(docker images --format "{{.Repository}}" 2>/dev/null | grep "l22-reranking-system-frontend" || echo "")

if [ -z "$BACKEND_IMAGE" ]; then
    echo "📦 Backend image not found. Building..."
    docker-compose build backend
else
    echo "✅ Backend image exists"
fi

if [ -z "$FRONTEND_IMAGE" ]; then
    echo "📦 Frontend image not found. Building..."
    docker-compose build frontend
else
    echo "✅ Frontend image exists"
fi

# Start services
echo "🚀 Starting all services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check health
echo "🏥 Checking service health..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend is healthy!"
        break
    fi
    echo "  Attempt $i/30..."
    sleep 2
done

# Show status
echo ""
echo "📊 Service Status:"
docker-compose ps

echo ""
echo "✅ Services started!"
echo "📊 Dashboard: http://localhost:3000"
echo "🔧 API: http://localhost:8000"
echo "📖 Docs: http://localhost:8000/docs"

