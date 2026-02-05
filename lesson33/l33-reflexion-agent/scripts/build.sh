#!/bin/bash
set -e

echo "Building L33 Reflexion Agent..."

# Backend
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Frontend
echo "Installing Node.js dependencies..."
cd frontend
npm install
cd ..

echo "✓ Build complete"
