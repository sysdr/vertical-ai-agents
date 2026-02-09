#!/bin/bash

echo "Stopping L36 PlannerAgent services..."

# Kill Python/Uvicorn
pkill -f "uvicorn app.main:app" 2>/dev/null || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# Kill Node/React
pkill -f "react-scripts start" 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

echo "Services stopped"
