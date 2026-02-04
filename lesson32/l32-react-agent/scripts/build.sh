#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Building L32 ReAct Agent..."

# Build Docker images
docker-compose build

# Install frontend dependencies
cd frontend
npm install
cd ..

echo "✓ Build complete"
