# Quick Start Guide

## Current Issue
The backend server is not running. Here's how to start it:

## Step 1: Set Up API Key

You need a Google Gemini API key. Get one from: https://makersuite.google.com/app/apikey

Then create the .env file:
```bash
cd backend
echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
```

Or use the interactive script:
```bash
./scripts/setup_api_key.sh
```

## Step 2: Start the Backend

```bash
cd backend
source venv/bin/activate
python main.py
```

Or use the start script (starts both backend and frontend):
```bash
./scripts/start.sh
```

## Step 3: Verify Backend is Running

In another terminal, check:
```bash
curl http://localhost:8000/health
```

You should see:
```json
{"status":"healthy","timestamp":"...","registered_tools":4}
```

## Step 4: Frontend

The frontend should already be running. If not:
```bash
cd frontend
npm run dev
```

Then open: http://localhost:3000

## Troubleshooting

- **500 errors**: Backend is not running or has errors
- **WebSocket errors**: Backend is not running (will auto-reconnect when backend starts)
- **API key errors**: Make sure .env file exists with valid API key
