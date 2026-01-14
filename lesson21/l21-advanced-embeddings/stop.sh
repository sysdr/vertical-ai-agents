#!/bin/bash
echo "🛑 Stopping L21..."

pkill -f "uvicorn app.main:app"
pkill -f "react-scripts start"

echo "✓ Stopped"
