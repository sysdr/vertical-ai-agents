# API Key Setup Instructions

## Problem
If you see this error:
```
Error: 400 API key expired. Please renew the API key.
```

## Solution

### Step 1: Get a New API Key

1. Visit: **https://aistudio.google.com/app/apikey**
2. Sign in with your Google account
3. Click **"Create API Key"** or use an existing key
4. Copy the API key (it starts with `AIza...`)

### Step 2: Update the API Key

**Option A: Using the update script (Recommended)**
```bash
./update_api_key.sh YOUR_NEW_API_KEY
```

**Option B: Manual update**
```bash
# Edit the .env file
nano conversational-agent/backend/.env

# Replace the GEMINI_API_KEY line with:
GEMINI_API_KEY=YOUR_NEW_API_KEY_HERE

# Save and exit (Ctrl+X, then Y, then Enter)
```

### Step 3: Restart the Backend

```bash
# Stop the backend
pkill -f "uvicorn api:app"

# Start it again
cd conversational-agent && bash scripts/start.sh &
```

Or use the convenience script:
```bash
./start.sh
```

### Step 4: Verify

Test the API:
```bash
curl http://localhost:8000/health
```

You should see: `{"status":"healthy"}`

## Quick Fix Script

If you have a new API key ready:
```bash
# Update and restart in one command
./update_api_key.sh YOUR_NEW_API_KEY
```

## Troubleshooting

- **Backend won't start**: Check logs with `tail -f /tmp/backend.log`
- **Still getting errors**: Make sure you restarted the backend after updating the key
- **Key format**: The key should start with `AIza` and be about 39 characters long




