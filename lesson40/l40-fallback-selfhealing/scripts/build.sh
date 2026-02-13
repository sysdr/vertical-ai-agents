#!/bin/bash
echo "Building L40 Fallback & Self-Healing..."

# Build Docker images
docker-compose build

# Install frontend dependencies
cd frontend
npm install
cd ..

echo "✅ Build complete!"
