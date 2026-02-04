#!/bin/bash
# Setup Gemini API key for L32 ReAct Agent
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f ".env" ]; then
    echo "✓ .env already exists"
    grep -q "GEMINI_API_KEY=" .env && echo "  GEMINI_API_KEY is set" || echo "  Add GEMINI_API_KEY=your_key to .env"
    exit 0
fi

if [ -f ".env.example" ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
    echo ""
    echo "IMPORTANT: Edit .env and add your Gemini API key:"
    echo "  1. Get a key at https://aistudio.google.com/apikey"
    echo "  2. Replace 'your_api_key_here' in .env with your key"
    echo ""
    echo "  nano .env   # or use your editor"
else
    echo "GEMINI_API_KEY=your_api_key_here" > .env
    echo "Created .env - add your key from https://aistudio.google.com/apikey"
fi
