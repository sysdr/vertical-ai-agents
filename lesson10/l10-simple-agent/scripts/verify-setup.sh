#!/bin/bash
echo "🔍 Verifying L10 Simple Agent Setup..."
echo ""

cd "$(dirname "$0")/.." || exit 1

# Check .env file
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "   Run: ./scripts/setup-api-key.sh"
    exit 1
fi

# Check API key
API_KEY=$(grep "^GEMINI_API_KEY=" .env 2>/dev/null | cut -d'=' -f2 | tr -d '"' | tr -d "'")
if [ -z "$API_KEY" ] || [ "$API_KEY" = "your_api_key_here" ]; then
    echo "❌ API key not set or using placeholder!"
    echo ""
    echo "   Current .env content:"
    grep "^GEMINI_API_KEY" .env
    echo ""
    echo "   To fix:"
    echo "   1. Get API key from: https://makersuite.google.com/app/apikey"
    echo "   2. Edit .env file: nano .env"
    echo "   3. Replace 'your_api_key_here' with your actual key"
    echo "   4. Restart: docker-compose restart backend"
    exit 1
fi

echo "✅ API key is set (${API_KEY:0:10}...)"

# Check Docker services
echo ""
echo "Checking Docker services..."
if ! docker-compose ps | grep -q "backend.*Up"; then
    echo "⚠️  Backend is not running. Starting..."
    docker-compose up -d backend
    sleep 3
fi

# Check if backend is responding
echo ""
echo "Testing backend API..."
if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo "✅ Backend is responding"
    
    # Test agent state
    STATE=$(curl -s http://localhost:8000/agent/state 2>/dev/null)
    if echo "$STATE" | grep -q "agent_id"; then
        echo "✅ Agent is initialized"
        echo ""
        echo "Agent State:"
        echo "$STATE" | python3 -m json.tool 2>/dev/null || echo "$STATE"
    else
        echo "⚠️  Agent state check failed"
    fi
else
    echo "❌ Backend is not responding"
    echo "   Check logs: docker-compose logs backend"
    exit 1
fi

# Check frontend
echo ""
echo "Checking frontend..."
if curl -s http://localhost:3000/ > /dev/null 2>&1; then
    echo "✅ Frontend is responding"
else
    echo "⚠️  Frontend is not responding"
fi

echo ""
echo "✅ Setup verification complete!"
echo ""
echo "📊 Dashboard: http://localhost:3000"
echo "🔌 API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"




