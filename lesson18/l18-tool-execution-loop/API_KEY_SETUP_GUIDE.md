# API Key Setup Guide

## ⚠️ Current Issue
The API key in the setup script has expired. You need to get your own Google Gemini API key.

## 🔑 Quick Setup (3 Steps)

### Step 1: Get Your API Key
1. Go to: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click **"Create API Key"** or use an existing key
4. Copy the API key (starts with `AIza...`)

### Step 2: Configure the API Key

**Option A: Use the setup script (Recommended)**
```bash
./scripts/setup_api_key.sh
```

**Option B: Manual setup**
```bash
cd backend
echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
```

### Step 3: Restart Backend
```bash
# Stop current backend
pkill -f "python main.py"

# Start backend
cd backend
source venv/bin/activate
python main.py
```

Or use the start script:
```bash
./scripts/start.sh
```

## ✅ Verify It Works

Test the API key:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in Tokyo?", "max_turns": 5}'
```

If you see a proper response (not an API key error), it's working!

## 🔍 Troubleshooting

### "API key expired" or "API_KEY_INVALID"
- The API key is invalid or expired
- Get a new key from https://makersuite.google.com/app/apikey
- Update the `.env` file and restart the backend

### "API Key not configured"
- The `.env` file doesn't exist or is empty
- Run `./scripts/setup_api_key.sh` to create it

### Backend won't start
- Check that the virtual environment is activated
- Check `backend.log` for error messages
- Verify Python dependencies are installed: `pip install -r requirements.txt`

## 📝 Notes

- The `.env` file is in `backend/.env` (not in the root)
- The API key should start with `AIza`
- Never commit your `.env` file to git (it's already in `.gitignore`)
- API keys are free but have usage limits


