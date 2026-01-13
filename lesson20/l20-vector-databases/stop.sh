#!/bin/bash
if command -v docker &> /dev/null; then
    echo "🛑 Stopping Docker containers..."
    docker-compose down
else
    echo "🛑 Stopping services..."
    pkill -f uvicorn
    pkill -f vite
fi
echo "✅ Services stopped"
