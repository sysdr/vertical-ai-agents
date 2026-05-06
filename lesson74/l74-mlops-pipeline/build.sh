#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "=== L74 Build ==="
if command -v docker &>/dev/null; then
  echo "Building Docker image..."
  docker build -t vaia-l74:latest .
  echo "Docker image built: vaia-l74:latest"
fi
echo "Building frontend..."
cd frontend && npm install --silent && npm run build
echo "Build complete."
