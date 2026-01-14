#!/bin/bash
cd "$(dirname "$0")/backend"
source venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"

# Check if critical packages are installed
python -c "import sentence_transformers" 2>/dev/null || {
    echo "Installing missing dependencies..."
    pip install sentence-transformers chromadb nltk google-generativeai tiktoken 2>&1 | tail -5
}

# Start the server
echo "Starting backend server on port 8000..."
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

