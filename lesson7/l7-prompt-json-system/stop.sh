#!/bin/bash

echo "🛑 Stopping L7 Prompt Engineering System..."

# Stop backend
pkill -f "uvicorn main:app"

# Stop frontend
pkill -f "react-scripts start"

echo "✅ System stopped"
