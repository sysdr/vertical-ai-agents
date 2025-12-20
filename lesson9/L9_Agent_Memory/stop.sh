#!/bin/bash
echo "Stopping L9 Agent Memory System..."

pkill -f "python api.py"
pkill -f "react-scripts start"

echo "✅ Stopped!"
