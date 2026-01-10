#!/bin/bash
echo "Stopping Naïve RAG system..."

if command -v docker &> /dev/null && [ -f "docker-compose.yml" ]; then
    docker-compose down
else
    if [ -f ".backend.pid" ]; then
        kill $(cat .backend.pid) 2>/dev/null
        rm .backend.pid
    fi
    if [ -f ".frontend.pid" ]; then
        kill $(cat .frontend.pid) 2>/dev/null
        rm .frontend.pid
    fi
fi

echo "✓ Services stopped"
