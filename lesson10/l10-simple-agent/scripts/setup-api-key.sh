#!/bin/bash
echo "🔑 Gemini API Key Setup"
echo ""

cd "$(dirname "$0")/.." || exit 1

if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << 'ENVEOF'
# Gemini API Key
# Get your API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_api_key_here
ENVEOF
fi

echo "Current .env file:"
cat .env
echo ""
echo "To set your API key:"
echo "1. Get your API key from: https://makersuite.google.com/app/apikey"
echo "2. Run: nano .env"
echo "3. Replace 'your_api_key_here' with your actual API key"
echo "4. Save and restart: docker-compose restart backend"
echo ""
read -p "Do you want to edit the .env file now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ${EDITOR:-nano} .env
    echo ""
    echo "✅ .env file updated!"
    echo "Restarting backend..."
    docker-compose restart backend
    echo "✅ Backend restarted!"
fi
