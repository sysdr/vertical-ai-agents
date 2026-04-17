#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "[stop] Stopping VAIA L64 stack..."
docker compose down
echo "[stop] Done."
