#!/bin/bash
echo "Stopping L30 RAG Observability..."

if command -v docker-compose &> /dev/null; then
    docker-compose down
else
    pkill -f "uvicorn app.main:app"
    pkill -f "react-scripts start"
fi

echo "Services stopped!"
