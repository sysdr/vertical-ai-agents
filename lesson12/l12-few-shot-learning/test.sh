#!/bin/bash
echo "🧪 Testing Few-Shot Learning System..."

# Wait for services
sleep 5

# Test backend health
echo "Testing backend..."
curl -s http://localhost:8000/health | grep -q "healthy" && echo "✅ Backend healthy" || echo "❌ Backend failed"

# Test classification
echo "Testing classification..."
curl -s -X POST http://localhost:8000/api/classify \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I want a refund",
    "task_description": "Classify customer message",
    "domain": "customer_support",
    "num_examples": 3
  }' | grep -q "classification" && echo "✅ Classification working" || echo "❌ Classification failed"

# Test examples
echo "Testing example retrieval..."
curl -s http://localhost:8000/api/examples | grep -q "customer_support" && echo "✅ Examples working" || echo "❌ Examples failed"

echo "✅ Tests complete!"
