#!/bin/bash
# Convenience script to stop all services from project root
echo "Stopping all services..."

# Stop backend
pkill -f "uvicorn api:app" 2>/dev/null
pkill -9 -f "uvicorn api:app" 2>/dev/null

# Stop frontend
pkill -f "react-scripts" 2>/dev/null
pkill -9 -f "react-scripts" 2>/dev/null
pkill -9 -f "node.*start.js" 2>/dev/null

sleep 1

# Verify
if ps aux | grep -E "(uvicorn|react-scripts)" | grep -v grep > /dev/null; then
    echo "⚠️  Some processes may still be running"
    ps aux | grep -E "(uvicorn|react-scripts)" | grep -v grep
else
    echo "✓ All services stopped successfully"
fi




