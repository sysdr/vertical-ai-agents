#!/bin/bash

echo "Stopping L6: LLM API Client..."

# Kill backend
pkill -f "uvicorn backend.main:app" || true

# Kill frontend
pkill -f "react-scripts start" || true

echo "✅ All services stopped"
