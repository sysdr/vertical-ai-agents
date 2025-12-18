# API Key Setup Instructions

## Problem
Your Gemini API key has been reported as leaked and is no longer valid.

## Solution: Get a New API Key

### Step 1: Get a New API Key

1. Go to Google AI Studio: https://makersuite.google.com/app/apikey
   - Or visit: https://aistudio.google.com/app/apikey

2. Sign in with your Google account

3. Click "Create API Key" or "Get API Key"

4. Copy the new API key

### Step 2: Update the API Key

**Option A: Update .env file directly**
```bash
cd /home/systemdrllp5/git/vertical-ai-agents/lesson7/l7-prompt-json-system/backend
nano .env
# Replace the GEMINI_API_KEY value with your new key
# Save and exit (Ctrl+X, then Y, then Enter)
```

**Option B: Use command line**
```bash
cd /home/systemdrllp5/git/vertical-ai-agents/lesson7/l7-prompt-json-system/backend
# Replace YOUR_NEW_API_KEY with your actual key
echo "GEMINI_API_KEY=YOUR_NEW_API_KEY" > .env
echo "ENVIRONMENT=development" >> .env
echo "LOG_LEVEL=INFO" >> .env
```

### Step 3: Restart the Backend

After updating the API key, restart the backend:

```bash
# Stop current backend
pkill -f "uvicorn main:app"

# Start backend again
cd /home/systemdrllp5/git/vertical-ai-agents/lesson7/l7-prompt-json-system/backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Step 4: Verify

Test that the API key works:
```bash
curl http://localhost:8000/health
```

Then try the dashboard again at http://localhost:3000

## Security Note

⚠️ **Important**: Never commit API keys to version control!
- The `.env` file should be in `.gitignore`
- Never share your API key publicly
- If a key is leaked, revoke it immediately and create a new one

## Alternative: Use Environment Variable

You can also set the API key as an environment variable instead of using .env:

```bash
export GEMINI_API_KEY=your_new_api_key_here
```

Then start the backend - it will use the environment variable.

