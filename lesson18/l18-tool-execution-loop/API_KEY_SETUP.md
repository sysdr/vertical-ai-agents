# 🔑 API Key Setup Guide

## Quick Setup (3 Methods)

### Method 1: Interactive Script (Easiest)
```bash
cd l18-tool-execution-loop
./scripts/setup_api_key.sh
```
Then follow the prompts and restart the backend.

### Method 2: Manual Setup
```bash
# 1. Get your API key from: https://makersuite.google.com/app/apikey
# 2. Create the .env file:
cd l18-tool-execution-loop/backend
echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
# 3. Restart backend (see below)
```

### Method 3: Edit Template File
```bash
cd l18-tool-execution-loop/backend
cp .env.example .env
# Then edit .env and replace "your_api_key_here" with your actual key
nano .env  # or use your preferred editor
```

## Restart Backend After Setup

**Option A: If backend is running in terminal**
- Press Ctrl+C to stop it
- Then run: `python main.py`

**Option B: If backend is running in background**
```bash
pkill -f "python main.py"
cd l18-tool-execution-loop/backend
source venv/bin/activate
python main.py
```

**Option C: Use start script**
```bash
cd l18-tool-execution-loop
./scripts/start.sh
```

## Verify It Works

After restarting, test the chat functionality in the frontend. 
The error should be gone and chat should work!

## Get Your API Key

1. Visit: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key" or use an existing one
4. Copy the key and use it in the setup above
