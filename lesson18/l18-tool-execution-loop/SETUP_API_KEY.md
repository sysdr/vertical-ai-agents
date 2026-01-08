# API Key Setup Instructions

## Problem
The API key in the code has expired. You need to set up your own Google Gemini API key.

## Solution

### Step 1: Get Your API Key
1. Go to https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key" or use an existing key
4. Copy your API key

### Step 2: Configure the API Key

**Option A: Using .env file (Recommended)**
```bash
cd backend
echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
```

**Option B: Using environment variable**
```bash
export GEMINI_API_KEY=your_actual_api_key_here
```

### Step 3: Restart the Backend
```bash
# Stop the current backend
pkill -f "python main.py"

# Restart it
cd backend
source venv/bin/activate
python main.py
```

## Verification
After restarting, the backend should start without errors. You can verify by:
```bash
curl http://localhost:8000/health
```

If you see a healthy response, the API key is working correctly.
