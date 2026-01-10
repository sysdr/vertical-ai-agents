#!/bin/bash
# Quick script to verify the API key is working

echo "Verifying API Key Configuration..."
echo "=================================="

# Check .env file
if [ -f .env ]; then
    API_KEY=$(grep GEMINI_API_KEY .env | cut -d'=' -f2)
    echo "✓ .env file found"
    echo "  API Key: ${API_KEY:0:20}...${API_KEY: -4}"
else
    echo "✗ .env file not found"
    exit 1
fi

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✓ Backend is running"
else
    echo "✗ Backend is not responding"
    exit 1
fi

# Test a simple query (requires documents to be uploaded first)
echo ""
echo "Testing query (requires documents in knowledge base)..."
RESPONSE=$(curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "test", "top_k": 1}')

if echo "$RESPONSE" | grep -q "API key expired\|API_KEY_INVALID\|expired"; then
    echo "✗ API key error detected"
    echo "  Response: $(echo "$RESPONSE" | head -c 200)"
    exit 1
elif echo "$RESPONSE" | grep -q "answer\|don't have enough information"; then
    echo "✓ API key is working! (Got valid response)"
    echo "  Response preview: $(echo "$RESPONSE" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('answer', d.get('detail', ''))[:100])" 2>/dev/null || echo 'Valid response')"
else
    echo "? Unexpected response (this might be OK if no documents are uploaded)"
    echo "  Response: $(echo "$RESPONSE" | head -c 200)"
fi

echo ""
echo "=================================="
echo "To test fully, upload a document and run: ./test_demo.sh"


