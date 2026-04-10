#!/usr/bin/env bash
set -e
echo "[L54] Building..."
cd "$(dirname "$0")/.."
python -m venv .venv
source .venv/bin/activate
pip install -q -r backend/requirements.txt
cd frontend && npm install --silent && cd ..
echo "[L54] Build complete"
