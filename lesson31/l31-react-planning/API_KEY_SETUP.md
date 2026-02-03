# 🔑 Gemini API Key Setup

The ReAct planning system uses Google's Gemini API. The API key in the project is expired. Configure your own key:

## Quick Setup

1. **Get a free API key**
   - Visit: https://aistudio.google.com/apikey
   - Sign in with your Google account
   - Click "Create API Key"
   - Copy the key

2. **Configure the key**

   **Option A – Docker (recommended)**  
   Create or edit `.env` in the project root (`l31-react-planning/`):

   ```bash
   echo "GEMINI_API_KEY=your_actual_key_here" > .env
   ```

   **Option B – Local run**  
   Edit `backend/.env`:

   ```bash
   GEMINI_API_KEY=your_actual_key_here
   ```

3. **Restart services**

   ```bash
   ./stop.sh && ./start.sh
   ```

## Verify

Run a planning task in the dashboard (http://localhost:3000). You should see real reasoning instead of "API key expired".

## Troubleshooting

| Error | Fix |
|-------|-----|
| API key expired / API_KEY_INVALID | Get a new key from https://aistudio.google.com/apikey |
| GEMINI_API_KEY not configured | Ensure `.env` exists and contains `GEMINI_API_KEY=...` |
| Backend won't start | Run from `l31-react-planning/` and check `backend/.env` or project root `.env` |

## Notes

- API keys are free for development
- Keys are stored in `.env` (add to `.gitignore` if needed)
- Docker uses the project root `.env`; local run uses `backend/.env`
