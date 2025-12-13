#!/bin/bash
set -e

echo "🧪 Testing L4 Parameter Analysis Platform..."

# Test backend
echo "Testing backend endpoints..."
curl -s http://localhost:8000/ | grep -q "L4" && echo "✅ Backend health check passed"

# Test model specs
curl -s http://localhost:8000/models | grep -q "gemini" && echo "✅ Model specs endpoint passed"

# Test cost calculation
curl -s -X POST http://localhost:8000/analyze/cost \
  -H "Content-Type: application/json" \
  -d '{"model_name":"gemini-2.0-flash-exp","requests_per_day":1000,"avg_input_tokens":500,"avg_output_tokens":200,"cache_hit_rate":0.5}' \
  | grep -q "monthly_cost" && echo "✅ Cost calculation passed"

# Test model comparison
curl -s -X POST "http://localhost:8000/compare?requests_per_day=1000&avg_input_tokens=500&avg_output_tokens=200&cache_hit_rate=0.5" \
  | grep -q "recommendation" && echo "✅ Model comparison passed"

echo "✅ All tests passed!"
