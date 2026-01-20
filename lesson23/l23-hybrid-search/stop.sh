#!/bin/bash
echo "Stopping L23: Hybrid Search System..."

# Kill processes
pkill -f "python main.py"
pkill -f "react-scripts start"
pkill -f "node.*react-scripts"

echo "✅ System stopped"
