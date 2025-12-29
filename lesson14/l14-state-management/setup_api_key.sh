#!/bin/bash

# Automated Gemini API Key Setup Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/backend/.env"

echo "🔑 Gemini API Key Setup"
echo "======================="
echo ""

# Check if .env exists
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Error: $ENV_FILE not found"
    exit 1
fi

echo "📋 Instructions:"
echo "1. We'll open Google AI Studio in your browser"
echo "2. Sign in with your Google account"
echo "3. Click 'Create API Key' or use an existing key"
echo "4. Copy the API key"
echo "5. Paste it here"
echo ""

# Try to open browser (works on most systems)
echo "🌐 Opening Google AI Studio..."
if command -v xdg-open > /dev/null; then
    xdg-open "https://aistudio.google.com/apikey" 2>/dev/null &
elif command -v open > /dev/null; then
    open "https://aistudio.google.com/apikey" 2>/dev/null &
elif command -v start > /dev/null; then
    start "https://aistudio.google.com/apikey" 2>/dev/null &
else
    echo "   Please manually visit: https://aistudio.google.com/apikey"
fi

sleep 2

echo ""
echo "📝 Paste your API key below (or press Ctrl+C to cancel):"
read -p "API Key: " NEW_KEY

if [ -z "$NEW_KEY" ]; then
    echo "❌ No API key provided. Exiting."
    exit 1
fi

# Remove any whitespace
NEW_KEY=$(echo "$NEW_KEY" | tr -d '[:space:]')

# Basic validation
if [ ${#NEW_KEY} -lt 30 ]; then
    echo "⚠️  Warning: API key seems too short (${#NEW_KEY} chars)."
    read -p "Continue anyway? (y/n): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "❌ Cancelled."
        exit 1
    fi
fi

# Update .env file
echo ""
echo "💾 Updating .env file..."

if grep -q "^GEMINI_API_KEY=" "$ENV_FILE"; then
    # Replace existing
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s|^GEMINI_API_KEY=.*|GEMINI_API_KEY=$NEW_KEY|" "$ENV_FILE"
    else
        sed -i "s|^GEMINI_API_KEY=.*|GEMINI_API_KEY=$NEW_KEY|" "$ENV_FILE"
    fi
    echo "✅ Updated existing API key"
else
    # Add new
    echo "GEMINI_API_KEY=$NEW_KEY" >> "$ENV_FILE"
    echo "✅ Added new API key"
fi

# Test the API key
echo ""
echo "🧪 Testing API key..."
cd "$SCRIPT_DIR/backend"
source venv/bin/activate 2>/dev/null || true

python3 << EOF
import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    print("❌ Error: API key not found in .env")
    sys.exit(1)

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content('Hello')
    print("✅ API key is valid! Test response:", response.text[:50] + "...")
except Exception as e:
    error_msg = str(e)
    if "API key" in error_msg or "expired" in error_msg.lower():
        print("❌ API key is invalid or expired:", error_msg[:100])
        sys.exit(1)
    else:
        print("⚠️  Warning:", error_msg[:100])
        print("   The key was saved, but there may be an issue.")
EOF

TEST_RESULT=$?

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo "🎉 Setup complete! The backend will auto-reload with the new key."
    echo ""
    echo "📌 Next steps:"
    echo "   - The backend should automatically reload (it has --reload flag)"
    echo "   - If not, restart with: ./stop.sh && ./start.sh"
    echo "   - Test it: curl -X POST http://localhost:8000/api/chat \\"
    echo "             -H 'Content-Type: application/json' \\"
    echo "             -d '{\"session_id\":\"test\",\"message\":\"hi\"}'"
else
    echo "⚠️  API key was saved but validation failed."
    echo "   Please check the key and try again, or restart the backend manually."
fi

