#!/bin/bash
echo "🔨 Building L10 Simple Agent..."

# Backend
cd backend
python -m venv venv 2>/dev/null || python3 -m venv venv
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install

echo "✅ Build complete!"
