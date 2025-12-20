#!/bin/bash
echo "Testing L9 Memory System..."

# Check backend health
echo "1. Checking backend health..."
curl -s http://localhost:8000/health | grep -q "healthy"
if [ $? -eq 0 ]; then
    echo "   ✅ Backend healthy"
else
    echo "   ❌ Backend not responding"
    exit 1
fi

# Test short-term storage
echo "2. Testing short-term memory storage..."
curl -s -X POST http://localhost:8000/api/store \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test_session","role":"user","content":"I like pizza"}' | grep -q "success"
if [ $? -eq 0 ]; then
    echo "   ✅ Short-term storage works"
else
    echo "   ❌ Short-term storage failed"
    exit 1
fi

# Test short-term recall
echo "3. Testing short-term memory recall..."
curl -s http://localhost:8000/api/recall/test_session | grep -q "pizza"
if [ $? -eq 0 ]; then
    echo "   ✅ Short-term recall works"
else
    echo "   ❌ Short-term recall failed"
    exit 1
fi

# Test long-term storage
echo "4. Testing long-term memory storage..."
curl -s -X POST http://localhost:8000/api/longterm/store \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test_user","fact":{"preference":"pizza","type":"food"}}' | grep -q "success"
if [ $? -eq 0 ]; then
    echo "   ✅ Long-term storage works"
else
    echo "   ❌ Long-term storage failed"
    exit 1
fi

# Test long-term search
echo "5. Testing long-term memory search..."
curl -s -X POST http://localhost:8000/api/longterm/search \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test_user","keywords":["pizza"],"limit":5}' | grep -q "preference"
if [ $? -eq 0 ]; then
    echo "   ✅ Long-term search works"
else
    echo "   ❌ Long-term search failed"
    exit 1
fi

# Test metrics
echo "6. Testing metrics endpoint..."
curl -s http://localhost:8000/api/metrics | grep -q "short_term_hits"
if [ $? -eq 0 ]; then
    echo "   ✅ Metrics endpoint works"
else
    echo "   ❌ Metrics endpoint failed"
    exit 1
fi

echo ""
echo "========================================="
echo "✅ All tests passed!"
echo "========================================="
echo "Open http://localhost:3000 to see dashboard"
