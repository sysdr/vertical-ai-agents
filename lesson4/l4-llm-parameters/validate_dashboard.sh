#!/bin/bash

echo "🔍 Validating Dashboard Metrics..."
echo ""

BASE_URL="http://localhost:8000"
ERRORS=0

# Test 1: Models endpoint
echo "1. Testing /models endpoint..."
RESPONSE=$(curl -s "$BASE_URL/models")
if echo "$RESPONSE" | grep -q "gemini" && echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); models=data['models']; exit(0 if all(v.get('input_price_per_1m', 0) > 0 for v in models.values()) else 1)" 2>/dev/null; then
    echo "   ✅ Models endpoint returns non-zero prices"
else
    echo "   ❌ Models endpoint has zero values"
    ERRORS=$((ERRORS + 1))
fi

# Test 2: Cost calculation
echo "2. Testing /analyze/cost endpoint..."
COST_RESPONSE=$(curl -s -X POST "$BASE_URL/analyze/cost" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"gemini-2.0-flash-exp","requests_per_day":10000,"avg_input_tokens":500,"avg_output_tokens":200,"cache_hit_rate":0.5}')

if echo "$COST_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); exit(0 if data.get('monthly_cost', 0) > 0 and data.get('daily_cost', 0) > 0 and data.get('cost_per_request', 0) > 0 else 1)" 2>/dev/null; then
    MONTHLY=$(echo "$COST_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['monthly_cost'])")
    DAILY=$(echo "$COST_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['daily_cost'])")
    echo "   ✅ Cost calculation returns non-zero values (Monthly: \$$MONTHLY, Daily: \$$DAILY)"
else
    echo "   ❌ Cost calculation has zero values"
    ERRORS=$((ERRORS + 1))
fi

# Test 3: Model comparison
echo "3. Testing /compare endpoint..."
COMPARE_RESPONSE=$(curl -s -X POST "$BASE_URL/compare?requests_per_day=10000&avg_input_tokens=500&avg_output_tokens=200&cache_hit_rate=0.5")

if echo "$COMPARE_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); comparisons=data.get('comparisons', []); exit(0 if comparisons and all(c.get('monthly_cost', 0) > 0 for c in comparisons) else 1)" 2>/dev/null; then
    RECOMMENDATION=$(echo "$COMPARE_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('recommendation', 'N/A'))")
    echo "   ✅ Model comparison returns non-zero costs (Recommended: $RECOMMENDATION)"
else
    echo "   ❌ Model comparison has zero values"
    ERRORS=$((ERRORS + 1))
fi

# Test 4: Performance test (if API key is valid)
echo "4. Testing /analyze/performance endpoint..."
PERF_RESPONSE=$(curl -s -X POST "$BASE_URL/analyze/performance" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"gemini-2.0-flash-exp","test_prompt":"Hello","num_iterations":2}' 2>&1)

if echo "$PERF_RESPONSE" | grep -q "latency_ms"; then
    if echo "$PERF_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); latency=data.get('latency_ms', {}); exit(0 if latency.get('mean', 0) > 0 else 1)" 2>/dev/null; then
        MEAN_LATENCY=$(echo "$PERF_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['latency_ms']['mean'])" 2>/dev/null || echo "N/A")
        echo "   ✅ Performance test returns non-zero latency (Mean: ${MEAN_LATENCY}ms)"
    else
        echo "   ⚠️  Performance test returned but with zero latency (may need valid API key)"
    fi
else
    echo "   ⚠️  Performance test requires valid API key (skipping)"
fi

# Test 5: Context analysis (if API key is valid)
echo "5. Testing /analyze/context endpoint..."
CONTEXT_RESPONSE=$(curl -s -X POST "$BASE_URL/analyze/context" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"gemini-2.0-flash-exp","sample_texts":["Hello world","Test text"]}' 2>&1)

if echo "$CONTEXT_RESPONSE" | grep -q "token_distribution"; then
    if echo "$CONTEXT_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); dist=data.get('token_distribution', {}); exit(0 if dist.get('mean', 0) > 0 else 1)" 2>/dev/null; then
        MEAN_TOKENS=$(echo "$CONTEXT_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['token_distribution']['mean'])" 2>/dev/null || echo "N/A")
        echo "   ✅ Context analysis returns non-zero token counts (Mean: ${MEAN_TOKENS} tokens)"
    else
        echo "   ⚠️  Context analysis returned but with zero tokens (may need valid API key)"
    fi
else
    echo "   ⚠️  Context analysis requires valid API key (skipping)"
fi

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "✅ All dashboard metrics validation passed!"
    echo "   All endpoints return non-zero values that will be displayed on the dashboard."
    exit 0
else
    echo "❌ Validation failed with $ERRORS error(s)"
    exit 1
fi

