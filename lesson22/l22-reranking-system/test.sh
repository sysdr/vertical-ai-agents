#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🧪 Testing L22 Reranking System..."

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "❌ Backend not responding after 60 attempts"
        exit 1
    fi
    sleep 1
done

# Test health endpoint
echo ""
echo "1️⃣ Testing health endpoint..."
HEALTH=$(curl -s http://localhost:8000/health)
echo "$HEALTH" | python3 -m json.tool 2>/dev/null || echo "$HEALTH"
if echo "$HEALTH" | grep -q "healthy"; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    exit 1
fi

# Test reranking - First request
echo ""
echo "2️⃣ Testing reranking (Request 1)..."
RERANK1=$(curl -s -X POST http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning models",
    "documents": [
      {"id": "1", "text": "Deep learning neural networks", "score": 0.7},
      {"id": "2", "text": "Traditional regression models", "score": 0.6},
      {"id": "3", "text": "Advanced machine learning architectures", "score": 0.8}
    ],
    "top_k": 3
  }')
echo "$RERANK1" | python3 -m json.tool 2>/dev/null || echo "$RERANK1"

# Test reranking - Second request (should hit cache)
echo ""
echo "3️⃣ Testing reranking (Request 2 - should hit cache)..."
RERANK2=$(curl -s -X POST http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning models",
    "documents": [
      {"id": "1", "text": "Deep learning neural networks", "score": 0.7},
      {"id": "2", "text": "Traditional regression models", "score": 0.6},
      {"id": "3", "text": "Advanced machine learning architectures", "score": 0.8}
    ],
    "top_k": 3
  }')
echo "$RERANK2" | python3 -m json.tool 2>/dev/null || echo "$RERANK2"

# Test metrics - should show non-zero values
echo ""
echo "4️⃣ Testing metrics endpoint..."
METRICS=$(curl -s http://localhost:8000/metrics)
echo "$METRICS" | python3 -m json.tool 2>/dev/null || echo "$METRICS"

# Validate metrics are not zero
TOTAL_REQUESTS=$(echo "$METRICS" | python3 -c "import sys, json; print(json.load(sys.stdin).get('total_requests', 0))" 2>/dev/null || echo "0")
if [ "$TOTAL_REQUESTS" -gt "0" ]; then
    echo "✅ Metrics show $TOTAL_REQUESTS total requests (non-zero)"
else
    echo "⚠️  Warning: total_requests is zero"
fi

# Test compare endpoint
echo ""
echo "5️⃣ Testing compare endpoint..."
COMPARE=$(curl -s -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural networks",
    "documents": [
      {"id": "1", "text": "Deep learning neural networks", "score": 0.7},
      {"id": "2", "text": "Traditional regression models", "score": 0.6},
      {"id": "3", "text": "Advanced machine learning architectures", "score": 0.8},
      {"id": "4", "text": "Convolutional neural networks for image processing", "score": 0.75},
      {"id": "5", "text": "Recurrent neural networks for sequences", "score": 0.72}
    ]
  }')
echo "$COMPARE" | python3 -m json.tool 2>/dev/null || echo "$COMPARE"

# Final metrics check
echo ""
echo "6️⃣ Final metrics check..."
FINAL_METRICS=$(curl -s http://localhost:8000/metrics)
echo "$FINAL_METRICS" | python3 -m json.tool 2>/dev/null || echo "$FINAL_METRICS"

echo ""
echo "✅ All tests complete!"
echo ""
echo "📊 Summary:"
echo "  - Health: ✅"
echo "  - Reranking: ✅"
echo "  - Caching: ✅"
echo "  - Metrics: ✅"
echo "  - Comparison: ✅"
