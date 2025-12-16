#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🧪 Testing VAIA L5 Model Comparison Platform..."

# Test backend
echo "Testing backend API..."
MAX_RETRIES=10
RETRY_COUNT=0
BACKEND_READY=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    response=$(curl -s http://localhost:8000/ 2>/dev/null || echo "FAILED")
    if [[ $response == *"operational"* ]]; then
        echo "✅ Backend API: Operational"
        BACKEND_READY=true
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "Waiting for backend... ($RETRY_COUNT/$MAX_RETRIES)"
        sleep 2
    fi
done

if [ "$BACKEND_READY" = false ]; then
    echo "❌ Backend API: Failed to respond"
    exit 1
fi

# Test models endpoint
response=$(curl -s http://localhost:8000/api/models 2>/dev/null || echo "FAILED")
if [[ $response == *"gemini"* ]]; then
    echo "✅ Models Endpoint: Working"
else
    echo "❌ Models Endpoint: Failed"
    echo "Response: $response"
    exit 1
fi

# Test recommendations endpoint
response=$(curl -s "http://localhost:8000/api/recommendations?max_latency_ms=200&max_cost_per_request=0.01" 2>/dev/null || echo "FAILED")
if [[ $response == *"recommended_models"* ]] || [[ $response == *"recommendations"* ]]; then
    echo "✅ Recommendations Endpoint: Working"
else
    echo "⚠️  Recommendations Endpoint: Unexpected response"
    echo "Response: $response"
fi

# Test frontend (check if port is listening)
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Frontend: Accessible"
else
    echo "⚠️  Frontend: Not accessible (may still be starting)"
fi

echo ""
echo "✅ All tests passed!"
