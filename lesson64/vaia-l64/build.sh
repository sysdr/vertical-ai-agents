#!/usr/bin/env bash
# Build the production runtime Docker image locally
set -euo pipefail
cd "$(dirname "$0")"
echo "[build] Building vaia-agent:local (runtime stage)..."
docker build --target runtime -t vaia-agent:local .
echo "[build] Image size:"
docker images vaia-agent:local --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
