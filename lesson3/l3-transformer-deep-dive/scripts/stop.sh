#!/bin/bash

echo "Stopping L3 Transformer Visualizer..."

if command -v docker-compose &> /dev/null; then
    docker-compose down
else
    # Kill processes
    pkill -f "uvicorn app.main:app"
    pkill -f "react-scripts start"
fi

echo "✅ Services stopped!"
