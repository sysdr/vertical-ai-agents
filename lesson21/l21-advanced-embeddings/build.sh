#!/bin/bash
echo "🔨 Building L21 project..."

# Backend setup
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
cd ..

# Frontend setup
cd frontend
npm install
cd ..

echo "✓ Build complete!"
