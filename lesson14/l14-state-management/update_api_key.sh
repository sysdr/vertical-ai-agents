#!/bin/bash

# Script to help update the Gemini API key

echo "🔑 Gemini API Key Update Helper"
echo "================================"
echo ""

# Check if .env file exists
ENV_FILE="backend/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Error: $ENV_FILE not found"
    exit 1
fi

echo "Current API key status:"
CURRENT_KEY=$(grep "GEMINI_API_KEY=" "$ENV_FILE" | cut -d'=' -f2)
if [ -z "$CURRENT_KEY" ]; then
    echo "  ⚠️  No API key found"
else
    echo "  Key: ${CURRENT_KEY:0:20}... (length: ${#CURRENT_KEY})"
fi

echo ""
echo "To get a new API key:"
echo "1. Visit: https://aistudio.google.com/apikey"
echo "2. Sign in with your Google account"
echo "3. Click 'Create API Key'"
echo "4. Copy the generated key"
echo ""

read -p "Enter your new Gemini API key (or press Enter to skip): " NEW_KEY

if [ -z "$NEW_KEY" ]; then
    echo "⏭️  Skipping API key update"
    exit 0
fi

# Validate key format (basic check)
if [ ${#NEW_KEY} -lt 30 ]; then
    echo "⚠️  Warning: API key seems too short. Are you sure it's correct?"
    read -p "Continue anyway? (y/n): " confirm
    if [ "$confirm" != "y" ]; then
        echo "❌ Update cancelled"
        exit 1
    fi
fi

# Update .env file
if grep -q "GEMINI_API_KEY=" "$ENV_FILE"; then
    # Replace existing key
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s|GEMINI_API_KEY=.*|GEMINI_API_KEY=$NEW_KEY|" "$ENV_FILE"
    else
        # Linux
        sed -i "s|GEMINI_API_KEY=.*|GEMINI_API_KEY=$NEW_KEY|" "$ENV_FILE"
    fi
else
    # Add new key
    echo "GEMINI_API_KEY=$NEW_KEY" >> "$ENV_FILE"
fi

echo "✅ API key updated successfully!"
echo ""
echo "The backend should auto-reload. If not, restart it with:"
echo "  ./stop.sh && ./start.sh"
echo ""
echo "Test the API key with:"
echo "  curl -X POST http://localhost:8000/api/chat -H 'Content-Type: application/json' -d '{\"session_id\":\"test\",\"message\":\"hi\"}'"

