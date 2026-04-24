#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "▶ Building Docker images…"
docker-compose build
echo "▶ Building React app…"
cd frontend && npm install --silent && npm run build
echo "✓ Build complete."
