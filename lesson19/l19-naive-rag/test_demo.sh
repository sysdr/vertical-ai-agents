#!/bin/bash
# Test script to validate dashboard metrics and demo functionality

API_BASE="http://localhost:8000"
echo "Testing Naïve RAG System Demo..."
echo "================================"

# Wait for backend to be ready
echo "Waiting for backend to be ready..."
for i in {1..30}; do
    if curl -s "$API_BASE/health" >/dev/null 2>&1; then
        echo "✓ Backend is ready!"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo "✗ Backend failed to start"
        exit 1
    fi
done

# Check initial stats
echo ""
echo "1. Checking initial stats..."
STATS=$(curl -s "$API_BASE/stats")
echo "$STATS" | python3 -m json.tool 2>/dev/null || echo "$STATS"

# Upload sample document
echo ""
echo "2. Uploading sample document..."
UPLOAD_RESPONSE=$(curl -s -X POST -F "file=@data/sample_doc1.txt" "$API_BASE/documents/upload")
echo "$UPLOAD_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$UPLOAD_RESPONSE"

# Upload second document
echo ""
echo "3. Uploading second document..."
UPLOAD_RESPONSE2=$(curl -s -X POST -F "file=@data/sample_doc2.txt" "$API_BASE/documents/upload")
echo "$UPLOAD_RESPONSE2" | python3 -m json.tool 2>/dev/null || echo "$UPLOAD_RESPONSE2"

# Check stats after upload
echo ""
echo "4. Checking stats after upload..."
STATS_AFTER=$(curl -s "$API_BASE/stats")
echo "$STATS_AFTER" | python3 -m json.tool 2>/dev/null || echo "$STATS_AFTER"

# Make a query
echo ""
echo "5. Making a query..."
QUERY_RESPONSE=$(curl -s -X POST "$API_BASE/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "top_k": 3}')
echo "$QUERY_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$QUERY_RESPONSE"

# Make another query
echo ""
echo "6. Making another query..."
QUERY_RESPONSE2=$(curl -s -X POST "$API_BASE/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are chunking strategies?", "top_k": 3}')
echo "$QUERY_RESPONSE2" | python3 -m json.tool 2>/dev/null || echo "$QUERY_RESPONSE2"

# Final stats check
echo ""
echo "7. Final stats check (should have non-zero values)..."
FINAL_STATS=$(curl -s "$API_BASE/stats")
echo "$FINAL_STATS" | python3 -m json.tool 2>/dev/null || echo "$FINAL_STATS"

# Validate metrics
echo ""
echo "8. Validating metrics..."
TOTAL_CHUNKS=$(echo "$FINAL_STATS" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['knowledge_base']['total_chunks'])" 2>/dev/null)
TOTAL_QUERIES=$(echo "$FINAL_STATS" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['knowledge_base']['total_queries'])" 2>/dev/null)

if [ -n "$TOTAL_CHUNKS" ] && [ "$TOTAL_CHUNKS" != "0" ] && [ "$TOTAL_CHUNKS" != "null" ]; then
    echo "✓ Total chunks: $TOTAL_CHUNKS (non-zero)"
else
    echo "✗ Total chunks is zero or missing"
    exit 1
fi

if [ -n "$TOTAL_QUERIES" ] && [ "$TOTAL_QUERIES" != "0" ] && [ "$TOTAL_QUERIES" != "null" ]; then
    echo "✓ Total queries: $TOTAL_QUERIES (non-zero)"
else
    echo "✗ Total queries is zero or missing"
    exit 1
fi

echo ""
echo "================================"
echo "✓ All tests passed! Dashboard metrics are updating correctly."


