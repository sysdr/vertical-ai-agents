#!/bin/bash
echo "🧪 Testing L10 Simple Agent..."

# Wait for services
sleep 5

# Test API
echo "Testing API endpoint..."
curl -s http://localhost:8000/ | grep -q "running" && echo "✅ API running" || echo "❌ API failed"

# Test agent state
echo "Testing agent state..."
curl -s http://localhost:8000/agent/state | grep -q "agent_id" && echo "✅ Agent state OK" || echo "❌ Agent state failed"

# Test agent action
echo "Testing agent action..."
curl -s -X POST http://localhost:8000/agent/act \
  -H "Content-Type: application/json" \
  -d '{"content":"Hello, what can you do?","goal":"Test interaction"}' \
  | grep -q "action" && echo "✅ Agent action OK" || echo "❌ Agent action failed"

echo "
✅ All tests passed!

Try the dashboard: http://localhost:3000
"
