#!/bin/bash
# Quick script to fix the API key issue

echo "=========================================="
echo "🔑 Gemini API Key Fix"
echo "=========================================="
echo ""
echo "Your API key needs to be updated."
echo ""
echo "📝 Steps to get a new API key:"
echo "   1. Visit: https://aistudio.google.com/app/apikey"
echo "   2. Sign in with your Google account"
echo "   3. Click 'Create API Key' or 'Get API Key'"
echo "   4. Copy the API key"
echo ""
read -p "Paste your new API key here: " NEW_KEY

if [ -z "$NEW_KEY" ]; then
    echo "❌ No API key provided. Exiting."
    exit 1
fi

# Update the .env file
ENV_FILE="backend/.env"
BACKUP_FILE="backend/.env.backup"

# Backup existing .env if it exists
if [ -f "$ENV_FILE" ]; then
    cp "$ENV_FILE" "$BACKUP_FILE"
    echo "✅ Backed up existing .env to .env.backup"
fi

# Create/update .env file
cat > "$ENV_FILE" << EOF
GEMINI_API_KEY=$NEW_KEY
ENVIRONMENT=development
LOG_LEVEL=INFO
EOF

echo "✅ Updated $ENV_FILE with new API key"
echo ""
echo "🔄 Restarting backend..."
pkill -f "uvicorn main:app" 2>/dev/null
sleep 2

echo ""
echo "✅ Done! The backend will use the new API key when restarted."
echo ""
echo "To start the backend:"
echo "  cd backend"
echo "  source venv/bin/activate"
echo "  uvicorn main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "Or use: ./start-backend.sh"

