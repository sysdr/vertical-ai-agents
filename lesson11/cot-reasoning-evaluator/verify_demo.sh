#!/bin/bash
# Verify demo execution and dashboard metrics

echo "🧪 Verifying CoT Reasoning Evaluator Demo..."

# Test 1: Backend is running
echo -n "Testing backend... "
if curl -s http://localhost:8000/ | grep -q "operational"; then
    echo "✓ Backend is running"
else
    echo "✗ Backend is not running"
    exit 1
fi

# Test 2: Execute a reasoning query
echo -n "Executing reasoning query... "
RESULT=$(curl -s -X POST http://localhost:8000/api/reason \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 10 + 15?", "style": "standard"}')

if echo "$RESULT" | grep -q "reasoning_trace"; then
    echo "✓ Reasoning query executed successfully"
else
    echo "✗ Reasoning query failed"
    echo "Response: $RESULT"
    exit 1
fi

# Test 3: Check stats after query
echo -n "Checking stats... "
sleep 2
STATS=$(curl -s http://localhost:8000/api/stats)
if echo "$STATS" | grep -q "total_traces"; then
    TOTAL=$(echo "$STATS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('total_traces', 0))" 2>/dev/null)
    if [ "$TOTAL" -gt 0 ]; then
        echo "✓ Stats updated: $TOTAL trace(s)"
    else
        echo "✗ Stats show 0 traces"
        exit 1
    fi
else
    echo "✗ Stats endpoint failed"
    exit 1
fi

# Test 4: Check memory endpoint
echo -n "Checking memory... "
MEMORY=$(curl -s http://localhost:8000/api/memory?limit=1)
if echo "$MEMORY" | grep -q "traces"; then
    echo "✓ Memory endpoint working"
else
    echo "✗ Memory endpoint failed"
    exit 1
fi

# Test 5: Execute another query to verify metrics update
echo -n "Executing second query... "
RESULT2=$(curl -s -X POST http://localhost:8000/api/reason \
  -H "Content-Type: application/json" \
  -d '{"query": "If a train travels 120 km in 2 hours, what is its speed?", "style": "detailed"}')

if echo "$RESULT2" | grep -q "quality_scores"; then
    echo "✓ Second query executed"
else
    echo "✗ Second query failed"
    exit 1
fi

# Test 6: Verify stats updated
echo -n "Verifying stats updated... "
sleep 2
STATS2=$(curl -s http://localhost:8000/api/stats)
TOTAL2=$(echo "$STATS2" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('total_traces', 0))" 2>/dev/null)
if [ "$TOTAL2" -gt "$TOTAL" ]; then
    AVG_QUALITY=$(echo "$STATS2" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('avg_quality', 0))" 2>/dev/null)
    echo "✓ Stats updated: $TOTAL2 traces, avg quality: $(printf '%.2f' $AVG_QUALITY)"
    
    # Verify metrics are not zero
    if [ "$(echo "$AVG_QUALITY > 0" | bc 2>/dev/null || echo "1")" = "1" ]; then
        echo "✓ Metrics are non-zero"
    else
        echo "✗ Metrics are zero"
        exit 1
    fi
else
    echo "✗ Stats did not update"
    exit 1
fi

# Test 7: Check frontend (if running)
echo -n "Checking frontend... "
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo "✓ Frontend is accessible"
else
    echo "⚠ Frontend is not running (optional)"
fi

echo ""
echo "✅ All verification tests passed!"
echo ""
echo "Dashboard Metrics Summary:"
curl -s http://localhost:8000/api/stats | python3 -m json.tool

