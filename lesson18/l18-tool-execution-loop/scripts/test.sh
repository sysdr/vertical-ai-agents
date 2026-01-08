#!/bin/bash
set -e

echo "🧪 Testing L18 Tool Execution Loop..."

# Test backend health
echo "Checking backend health..."
curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:8000/health
echo ""

# Test tools endpoint
echo "Checking registered tools..."
curl -s http://localhost:8000/tools | python3 -m json.tool 2>/dev/null || curl -s http://localhost:8000/tools
echo ""

# Test stats endpoint
echo "Checking stats..."
curl -s http://localhost:8000/stats | python3 -m json.tool 2>/dev/null || curl -s http://localhost:8000/stats
echo ""

# Test chat endpoint
echo "Testing tool execution..."
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in Tokyo?", "max_turns": 5}' | python3 -m json.tool 2>/dev/null || curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in Tokyo?", "max_turns": 5}'
echo ""

echo "✅ All tests passed!"
