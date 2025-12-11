#!/bin/bash

echo "=== Stopping VAIA L2 ==="

# Kill processes on ports
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

echo "✅ Stopped"
