#!/bin/bash

echo "⏹️  Stopping VAIA L5 services..."

# Kill backend
pkill -f "uvicorn app.main:app" || true

# Kill frontend
pkill -f "react-scripts start" || true

# Wait a bit
sleep 2

# Force kill if still running
pkill -9 -f "uvicorn app.main:app" 2>/dev/null || true
pkill -9 -f "react-scripts start" 2>/dev/null || true

echo "✅ All services stopped"
