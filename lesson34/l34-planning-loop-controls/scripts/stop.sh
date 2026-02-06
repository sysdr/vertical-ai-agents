#!/bin/bash

# Change to project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🛑 Stopping L34 services..."

if command -v docker &> /dev/null && docker-compose version &> /dev/null; then
    docker-compose down
    echo "✅ Docker services stopped"
else
    pkill -f "python main.py" || true
    pkill -f "react-scripts start" || true
    echo "✅ Services stopped"
fi
