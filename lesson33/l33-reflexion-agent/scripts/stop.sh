#!/bin/bash

echo "Stopping L33 Reflexion Agent..."

# Kill backend
pkill -f "backend.api" || true

# Kill frontend
pkill -f "react-scripts" || true

echo "✓ Services stopped"
