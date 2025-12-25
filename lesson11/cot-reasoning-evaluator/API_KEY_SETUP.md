# API Key Setup Guide

## Quick Setup

1. **Get your Gemini API key:**
   - Visit: https://makersuite.google.com/app/apikey
   - Sign in with your Google account
   - Click "Create API Key"
   - Copy the generated API key

2. **Create the .env file:**
   ```bash
   cd /home/systemdrllp5/git/vertical-ai-agents/lesson11/cot-reasoning-evaluator
   cp .env.template .env
   ```

3. **Edit the .env file and add your API key:**
   ```bash
   nano .env
   # Replace "your-api-key-here" with your actual API key
   ```

4. **Restart the backend server:**
   ```bash
   pkill -f "uvicorn main:app"
   cd backend
   nohup ./start_server.sh > /tmp/backend.log 2>&1 &
   ```

5. **Verify it's working:**
   - Refresh your browser
   - Try submitting a query again
   - The error should be gone and reasoning should work

## Alternative: Set environment variable directly

If you prefer not to use a .env file, you can export it in your shell:

```bash
export GEMINI_API_KEY="your-api-key-here"
pkill -f "uvicorn main:app"
cd /home/systemdrllp5/git/vertical-ai-agents/lesson11/cot-reasoning-evaluator/backend
GEMINI_API_KEY="your-api-key-here" nohup ./start_server.sh > /tmp/backend.log 2>&1 &
```

## Troubleshooting

- **Error: "API key was reported as leaked"**: The API key in the setup script is invalid. You must use your own API key.
- **Error: "403 Forbidden"**: Check that your API key is correct and has not been revoked.
- **Error: "Quota exceeded"**: You've reached your API usage limit. Check your Google Cloud Console.

