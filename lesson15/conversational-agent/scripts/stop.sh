#!/bin/bash
echo "Stopping Conversational Agent..."
pkill -f "uvicorn api:app"
pkill -f "react-scripts"
echo "✓ Stopped"
