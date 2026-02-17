#!/bin/bash

echo "Stopping VAIA L41 services..."

pkill -f "python.*main.py" || true
pkill -f "react-scripts" || true
pkill -f "node.*react-scripts" || true

echo "✅ All services stopped"
