#!/bin/bash
set -e

echo "🐳 Building Docker containers..."

# IMPORTANT: Set your API key as environment variable or update this line
# export GEMINI_API_KEY="YOUR_API_KEY_HERE"
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  Warning: GEMINI_API_KEY not set. Please set it before building."
    echo "   export GEMINI_API_KEY=your_api_key_here"
fi

cd docker
docker-compose build
echo "✅ Docker build complete!"
echo ""
echo "To start:"
echo "  cd docker && docker-compose up"
