#!/bin/bash
# Script to update the Gemini API key in .env file

if [ -z "$1" ]; then
    echo "Usage: ./update_api_key.sh <your_gemini_api_key>"
    echo ""
    echo "To get your API key, visit: https://makersuite.google.com/app/apikey"
    exit 1
fi

API_KEY="$1"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
fi

# Update API key in .env file
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/^GEMINI_API_KEY=.*/GEMINI_API_KEY=$API_KEY/" .env
else
    # Linux
    sed -i "s/^GEMINI_API_KEY=.*/GEMINI_API_KEY=$API_KEY/" .env
fi

echo "✓ API key updated in .env file"
echo ""
echo "To apply the changes, restart the services:"
echo "  docker-compose restart backend"
echo "  # OR for local deployment:"
echo "  ./stop.sh && ./start.sh"


