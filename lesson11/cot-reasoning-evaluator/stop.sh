#!/bin/bash

echo "🛑 Stopping L11 CoT Reasoning Evaluator..."

if command -v docker &> /dev/null && docker ps &> /dev/null; then
    if command -v docker-compose &> /dev/null; then
        docker-compose down
    elif docker compose version &> /dev/null; then
        docker compose down
    else
        pkill -f "uvicorn main:app"
        pkill -f "react-scripts start"
    fi
else
    pkill -f "uvicorn main:app"
    pkill -f "react-scripts start"
fi

echo "✅ Services stopped"
