#!/bin/bash
echo "🧪 Testing L31 ReAct Planning System..."

# Wait for services
sleep 5

# Test backend health
echo "Testing backend..."
HEALTH=$(curl -s http://localhost:8000/api/v1/health)
if echo "$HEALTH" | grep -q "healthy"; then
    echo "✅ Backend healthy"
else
    echo "❌ Backend not responding"
    exit 1
fi

# Test planning endpoint
echo "Testing planning endpoint..."
RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/plan \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Analyze system performance and identify bottlenecks",
    "max_iterations": 3
  }')

if echo "$RESPONSE" | grep -q "task_id"; then
    echo "✅ Planning endpoint working"
    echo "Sample response:"
    echo "$RESPONSE" | python3 -m json.tool | head -20
else
    echo "❌ Planning endpoint failed"
    exit 1
fi

# Test frontend
echo "Testing frontend..."
FRONTEND=$(curl -s http://localhost:3000)
if echo "$FRONTEND" | grep -q "root"; then
    echo "✅ Frontend accessible"
else
    echo "❌ Frontend not responding"
    exit 1
fi

echo ""
echo "✅ All tests passed!"
echo "🌐 Dashboard: http://localhost:3000"
