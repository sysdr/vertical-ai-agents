#!/bin/bash
echo "Stopping L29 services..."
pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
echo "✓ Services stopped"
