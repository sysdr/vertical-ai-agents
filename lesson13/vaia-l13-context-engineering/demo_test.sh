#!/bin/bash
# Demo script to test API endpoints and validate dashboard metrics

echo "=================================="
echo "VAIA L13: Demo Test Script"
echo "=================================="

API_URL="http://localhost:8000/api/v1"

# Test 1: Health check
echo ""
echo "1. Testing health endpoint..."
HEALTH=$(curl -s http://localhost:8000/health)
echo "Health: $HEALTH"

# Test 2: Token counting
echo ""
echo "2. Testing token counting..."
TOKEN_RESPONSE=$(curl -s -X POST "$API_URL/count-tokens" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a comprehensive test of the token counting functionality. It should analyze the text and provide token usage statistics.",
    "max_tokens": 30000
  }')
echo "Token Count Response:"
echo "$TOKEN_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$TOKEN_RESPONSE"

# Test 3: Summarization
echo ""
echo "3. Testing summarization..."
SUMMARY_RESPONSE=$(curl -s -X POST "$API_URL/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Context engineering is a critical aspect of building production AI agents. It involves managing token limits, optimizing context windows, and ensuring efficient use of language model APIs. This requires careful consideration of compression strategies, cost optimization, and quality preservation.",
    "strategy": "extractive",
    "target_ratio": 0.5
  }')
echo "Summary Response:"
echo "$SUMMARY_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$SUMMARY_RESPONSE"

# Test 4: Context optimization
echo ""
echo "4. Testing context optimization..."
OPTIMIZE_RESPONSE=$(curl -s -X POST "$API_URL/optimize-context" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a very long context that needs optimization. It contains extensive information that must be compressed to fit within token limits. The optimization process should intelligently reduce the size while maintaining key information and technical details.",
    "max_tokens": 100,
    "preserve_quality": true
  }')
echo "Optimize Response:"
echo "$OPTIMIZE_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$OPTIMIZE_RESPONSE"

# Test 5: Multiple optimizations to update dashboard
echo ""
echo "5. Running multiple optimizations to update dashboard metrics..."
for i in {1..3}; do
  echo "  Optimization $i..."
  curl -s -X POST "$API_URL/optimize-context" \
    -H "Content-Type: application/json" \
    -d "{
      \"text\": \"Test optimization $i: This text will be optimized to demonstrate the dashboard metrics update functionality.\",
      \"max_tokens\": 50,
      \"preserve_quality\": true
    }" > /dev/null
  sleep 1
done

echo ""
echo "=================================="
echo "✓ Demo tests complete!"
echo "=================================="
echo ""
echo "Dashboard should now show updated metrics:"
echo "- Total Optimizations: Should be > 0"
echo "- Tokens Saved: Should be > 0"
echo "- Average Compression: Should be > 0"
echo "- Cost Savings: Should be > 0"
echo ""
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"

