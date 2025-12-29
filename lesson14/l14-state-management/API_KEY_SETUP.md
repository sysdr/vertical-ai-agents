# 🔑 Gemini API Key Setup Guide

## Quick Setup (Recommended)

Run the automated setup script:

```bash
./setup_api_key.sh
```

This script will:
- Open Google AI Studio in your browser
- Guide you through getting your API key
- Automatically update your `.env` file
- Test the API key to ensure it works
- The backend will auto-reload with the new key

## Manual Setup

1. **Get Your API Key**
   - Visit: https://aistudio.google.com/apikey
   - Sign in with your Google account
   - Click "Create API Key"
   - Copy the generated key

2. **Update the .env File**
   ```bash
   cd backend
   # Edit .env and update this line:
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Restart the Backend** (if auto-reload doesn't work)
   ```bash
   ./stop.sh && ./start.sh
   ```

## Verify It Works

Test the API key:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test","message":"hi"}'
```

You should get a real AI response instead of a demo message.

## Troubleshooting

### "API key expired" error
- Your API key may have expired
- Get a new one from https://aistudio.google.com/apikey
- Update it using `./setup_api_key.sh`

### "API quota exceeded" error
- You've hit your usage limit
- Check your quota at https://aistudio.google.com/apikey
- Wait for quota reset or upgrade your plan

### Backend not reloading
- The backend uses `--reload` flag for auto-reload
- If changes don't apply, manually restart: `./stop.sh && ./start.sh`

## Notes

- API keys are **free** for development use
- Keys are stored in `backend/.env` (not committed to git)
- The system works in demo mode without a key, but responses are limited

