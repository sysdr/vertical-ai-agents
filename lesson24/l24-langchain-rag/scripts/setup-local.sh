#!/bin/bash
echo "Setting up L24 LangChain RAG (non-Docker)..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..

echo "Local setup complete!"
echo "To start:"
echo "1. Backend: source venv/bin/activate && python backend/main.py"
echo "2. Frontend: cd frontend && npm start"
