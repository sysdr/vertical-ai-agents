#!/bin/bash
echo "Building L9 Agent Memory System..."

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate
pip install -r requirements.txt
cd ..

# Frontend setup
cd frontend
npm install
cd ..

echo "✅ Build complete!"
