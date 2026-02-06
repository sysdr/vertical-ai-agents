#!/bin/bash
set -e

# Change to project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🧪 Testing L34 Planning Loop Controls..."

# Wait for services
echo "⏳ Waiting for services to start..."
sleep 5

# Test health endpoint
echo "Testing backend health..."
curl -f http://localhost:8000/health || {
    echo "❌ Backend health check failed"
    exit 1
}

# Test policies endpoint
echo "Testing policies endpoint..."
curl -f http://localhost:8000/api/policies || {
    echo "❌ Policies endpoint failed"
    exit 1
}

# Test agent execution
echo "Testing agent execution..."
curl -f -X POST http://localhost:8000/api/agent/execute \
    -H "Content-Type: application/json" \
    -d '{"task":"Calculate 2+2 and explain the result", "environment":"development"}' || {
    echo "❌ Agent execution failed"
    exit 1
}

# Test metrics
echo "Testing metrics endpoint..."
curl -f http://localhost:8000/api/metrics/summary || {
    echo "❌ Metrics endpoint failed"
    exit 1
}

echo ""
echo "✅ All tests passed!"
echo "🌐 Open http://localhost:3000 to view the dashboard"
