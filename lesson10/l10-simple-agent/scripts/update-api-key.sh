#!/bin/bash
# Quick script to update the Gemini API key

cd "$(dirname "$0")/.." || exit 1

echo "🔑 Gemini API Key Update"
echo ""
echo "Current .env file:"
cat .env
echo ""
echo ""

if [ -z "$1" ]; then
    echo "Usage: $0 <your_api_key>"
    echo ""
    echo "Example:"
    echo "  $0 AIzaSyDGswqDT4wQw_bd4WZtIgYAawRDZ0Gisn8"
    echo ""
    echo "To get your API key:"
    echo "  1. Visit: https://makersuite.google.com/app/apikey"
    echo "  2. Sign in with your Google account"
    echo "  3. Create or copy your API key"
    echo "  4. Run this script with your key"
    exit 1
fi

API_KEY="$1"

# Update .env file
cat > .env << EOF
# Gemini API Key
# Get your API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=$API_KEY
EOF

echo "✅ .env file updated with new API key"
echo ""
echo "Restarting backend container..."
docker-compose restart backend

echo ""
echo "✅ Backend restarted!"
echo ""
echo "The API key has been updated. The backend should now work correctly."
echo "You can test it by sending a message through the dashboard at http://localhost:3000"

