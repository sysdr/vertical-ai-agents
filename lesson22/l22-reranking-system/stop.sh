#!/bin/bash
echo "🛑 Stopping L22 Reranking System..."

if command -v docker &> /dev/null; then
    docker-compose down
else
    pkill -f "python main.py"
    pkill -f "npm start"
    redis-cli shutdown 2>/dev/null || true
fi

echo "✅ Services stopped"
