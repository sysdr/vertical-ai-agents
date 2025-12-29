#!/bin/bash

echo "🛑 Stopping L14 State Management..."

# Stop processes
pkill -f "uvicorn app.main:app" || true
pkill -f "npm start" || true

# Stop Docker containers
docker stop l14-postgres l14-redis || true
docker rm l14-postgres l14-redis || true

echo "✅ System stopped"
