#!/bin/bash
echo "Stopping VAIA L13: Context Engineering..."

if command -v docker &> /dev/null && [ -f "docker/docker-compose.yml" ]; then
    cd docker
    docker-compose down
    echo "✓ Docker services stopped"
else
    pkill -f "uvicorn main:app"
    pkill -f "react-scripts start"
    echo "✓ Local services stopped"
fi
