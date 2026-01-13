#!/bin/bash
# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🧪 Testing L20 Vector Database..."

# Wait for backend to be ready
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend is ready"
        break
    fi
    attempt=$((attempt + 1))
    sleep 1
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ Backend not responding. Is it running?"
    exit 1
fi

# Test indexing
echo "📄 Testing indexing..."
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"text":"Machine learning is a subset of artificial intelligence focused on algorithms that learn from data.","source":"ml-basics.txt","collection":"default"}'

echo -e "\n🔍 Testing search..."
# Test search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What is machine learning?","collection":"default","n_results":3}'

echo -e "\n📊 Testing stats..."
# Test stats
curl http://localhost:8000/stats/default

echo -e "\n✅ Tests complete - Check http://localhost:3000 for dashboard"
