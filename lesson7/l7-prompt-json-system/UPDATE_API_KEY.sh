#!/bin/bash
# Script to update the Gemini API key

echo "🔑 Gemini API Key Update Script"
echo "================================"
echo ""
echo "Your current API key has been disabled (leaked)."
echo ""
echo "To get a new API key:"
echo "1. Visit: https://aistudio.google.com/app/apikey"
echo "2. Sign in with your Google account"
echo "3. Click 'Create API Key' or 'Get API Key'"
echo "4. Copy the new API key"
echo ""
read -p "Enter your new Gemini API key: " NEW_API_KEY

if [ -z "$NEW_API_KEY" ]; then
    echo "❌ No API key provided. Exiting."
    exit 1
fi

# Update .env file
ENV_FILE="backend/.env"
if [ -f "$ENV_FILE" ]; then
    # Update the API key in .env
    if grep -q "GEMINI_API_KEY=" "$ENV_FILE"; then
        # Use sed to replace the API key (works on Linux)
        sed -i "s|GEMINI_API_KEY=.*|GEMINI_API_KEY=$NEW_API_KEY|" "$ENV_FILE"
        echo "✅ Updated $ENV_FILE"
    else
        echo "GEMINI_API_KEY=$NEW_API_KEY" >> "$ENV_FILE"
        echo "✅ Added API key to $ENV_FILE"
    fi
else
    # Create .env file if it doesn't exist
    cat > "$ENV_FILE" << EOF
GEMINI_API_KEY=$NEW_API_KEY
ENVIRONMENT=development
LOG_LEVEL=INFO
EOF
    echo "✅ Created $ENV_FILE with new API key"
fi

echo ""
echo "✅ API key updated successfully!"
echo ""
echo "Next steps:"
echo "1. Restart the backend server"
echo "2. Test the dashboard at http://localhost:3000"
echo ""

