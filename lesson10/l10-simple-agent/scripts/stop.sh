#!/bin/bash
echo "🛑 Stopping L10 Simple Agent..."

if command -v docker &> /dev/null; then
    docker-compose down
else
    pkill -f "python main.py"
    pkill -f "react-scripts start"
fi

echo "✅ Services stopped!"
