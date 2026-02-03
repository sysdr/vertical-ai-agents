#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
echo "Building L30 RAG Observability..."

# Install frontend dependencies first (before Docker, to avoid root-owned files)
(cd frontend && npm install) || echo "Frontend npm install skipped (optional when using Docker)"

# Build Docker images
docker-compose build

echo "Build complete!"
