#!/bin/bash

# IMPORTANT: Set your API key as environment variable or update this line
# export GEMINI_API_KEY="YOUR_API_KEY_HERE"
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  Warning: GEMINI_API_KEY not set. Please set it before starting."
    echo "   export GEMINI_API_KEY=your_api_key_here"
    exit 1
fi

echo "🐳 Starting Docker containers..."
cd docker
docker-compose up
