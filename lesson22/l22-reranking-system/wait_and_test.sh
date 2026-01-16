#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "⏳ Waiting for backend build to complete..."

# Wait for backend image
MAX_WAIT=1800  # 30 minutes
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if docker images | grep -q "l22-reranking-system-backend"; then
        echo "✅ Backend image ready!"
        break
    fi
    echo "  Still building... ($ELAPSED seconds elapsed)"
    sleep 30
    ELAPSED=$((ELAPSED + 30))
done

if ! docker images | grep -q "l22-reranking-system-backend"; then
    echo "❌ Backend build timeout after $MAX_WAIT seconds"
    exit 1
fi

# Start services
echo "🚀 Starting all services..."
docker-compose down 2>/dev/null || true
docker-compose up -d

# Wait for services
echo "⏳ Waiting for services to be ready..."
sleep 15

# Check health
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend is healthy!"
        break
    fi
    sleep 2
done

# Run tests
echo ""
echo "🧪 Running tests..."
bash test.sh

echo ""
echo "✅ All done!"

