#!/bin/bash

echo "🛑 Stopping services..."

# Kill processes
pkill -f "python main.py" || true
pkill -f "vite" || true

echo "✅ Services stopped"
