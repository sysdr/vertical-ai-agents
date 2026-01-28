#!/bin/bash
echo "Building L26 Multi-Turn RAG System..."

# Backend dependencies
echo "Installing backend dependencies..."
cd backend
pip3 install --user --break-system-packages --upgrade pip
pip3 install --user --break-system-packages -r requirements.txt
cd ..

# Frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo "✅ Build complete!"
