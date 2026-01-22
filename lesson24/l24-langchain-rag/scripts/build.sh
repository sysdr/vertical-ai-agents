#!/bin/bash
echo "Building L24 LangChain RAG..."

# Build Docker images
docker-compose build

# Install frontend dependencies
cd frontend && npm install && cd ..

echo "Build complete!"
