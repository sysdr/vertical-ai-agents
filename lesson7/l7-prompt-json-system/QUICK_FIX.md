# Quick Fix: API Key Issue

## Current Problem
Your Gemini API key is invalid or has been disabled. You need to get a new one.

## Solution (3 Steps)

### Step 1: Get a New API Key

1. **Visit Google AI Studio**: https://aistudio.google.com/app/apikey
   - Alternative: https://makersuite.google.com/app/apikey

2. **Sign in** with your Google account

3. **Create API Key**:
   - Click "Create API Key" or "Get API Key"
   - Select a project (or create a new one)
   - Copy the generated API key

### Step 2: Update the API Key

**Option A: Use the fix script (Interactive)**
```bash
cd /home/systemdrllp5/git/vertical-ai-agents/lesson7/l7-prompt-json-system
./fix_api_key.sh
# When prompted, paste your new API key
```

**Option B: Manual Update**
```bash
cd /home/systemdrllp5/git/vertical-ai-agents/lesson7/l7-prompt-json-system/backend
nano .env
# Replace the GEMINI_API_KEY value with your new key
# Save: Ctrl+X, then Y, then Enter
```

**Option C: One-line command**
```bash
cd /home/systemdrllp5/git/vertical-ai-agents/lesson7/l7-prompt-json-system/backend
# Replace YOUR_NEW_KEY with your actual API key
echo "GEMINI_API_KEY=YOUR_NEW_KEY" > .env
echo "ENVIRONMENT=development" >> .env
echo "LOG_LEVEL=INFO" >> .env
```

### Step 3: Restart Backend

```bash
# Stop current backend
pkill -f "uvicorn main:app"

# Start backend
cd /home/systemdrllp5/git/vertical-ai-agents/lesson7/l7-prompt-json-system/backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or use the start script:
```bash
cd /home/systemdrllp5/git/vertical-ai-agents/lesson7/l7-prompt-json-system
./start-backend.sh
```

## Verify It Works

1. **Check health endpoint**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Test the dashboard**: http://localhost:3000
   - Click "Test: User Profile" button
   - Should see successful parse results

## Troubleshooting

**If you still get API key errors:**
- Make sure you copied the entire API key (they're long)
- Check that there are no extra spaces in the .env file
- Verify the .env file is in: `backend/.env`
- Restart the backend after updating the key

**If you can't get an API key:**
- Make sure you're signed in to Google
- Check that Gemini API is enabled in your Google Cloud project
- Try a different Google account if needed

