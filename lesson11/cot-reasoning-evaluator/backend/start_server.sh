#!/bin/bash
# Load .env file if it exists
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$PROJECT_DIR/.env" ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

# Check if API key is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  WARNING: GEMINI_API_KEY is not set!"
    echo ""
    echo "Please set your Gemini API key:"
    echo "  1. Create a .env file in the project root: $PROJECT_DIR/.env"
    echo "  2. Add: GEMINI_API_KEY=your-api-key-here"
    echo "  3. Get an API key from: https://makersuite.google.com/app/apikey"
    echo ""
    echo "The server will start but the /api/reason endpoint will not work without a valid API key."
    echo ""
fi

cd "$SCRIPT_DIR"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

