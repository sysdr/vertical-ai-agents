#!/bin/bash
echo "Stopping L40 Fallback & Self-Healing..."

if command -v docker &> /dev/null; then
    docker-compose down
else
    pkill -f "uvicorn main:app"
    pkill -f "npm start"
fi

echo "✅ Services stopped!"
