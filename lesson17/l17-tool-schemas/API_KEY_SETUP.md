# Gemini API Key Setup

## Quick Fix

If you're seeing the error: "The Gemini API key is expired or invalid"

### Option 1: Use the helper script (Recommended)
```bash
./update_api_key.sh
```

### Option 2: Manual update
1. Get a new API key from: https://makersuite.google.com/app/apikey
2. Edit `backend/.env` and update the `GEMINI_API_KEY` value
3. Restart the backend: `./stop.sh && ./start.sh`

## Getting a New API Key

1. Visit: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key" or select an existing project
4. Copy the generated API key
5. Use the `update_api_key.sh` script or manually update `backend/.env`

## Verify API Key

After updating, test the endpoint:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the weather in Paris?"}'
```

If successful, you should see a response with tool calls and results.


