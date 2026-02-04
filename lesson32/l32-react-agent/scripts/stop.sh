#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Stopping L32 ReAct Agent..."

if command -v docker-compose &> /dev/null; then
    docker-compose down
else
    pkill -f "python app.py"
    pkill -f "npm start"
fi

echo "✓ Services stopped"
