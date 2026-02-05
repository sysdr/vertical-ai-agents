# API Key Setup

The L33 Reflexion Agent uses Google's Gemini API. You need a valid API key to run tasks.

## Quick Setup

1. **Get an API key** from [Google AI Studio](https://aistudio.google.com/apikey)

2. **Edit `.env`** and add or update:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

3. **Restart the app**:
   ```bash
   ./scripts/stop.sh
   ./scripts/start.sh
   ```

## Troubleshooting

- **"API key expired"** – Your key has expired. Generate a new one at the link above.
- **"No API_KEY found"** – Ensure `.env` exists and contains `GEMINI_API_KEY=...`
- **Dashboard shows API key warning** – The health check detected a missing or placeholder key.
- **"Quota exceeded" / 429** – Free tier: ~5 requests/min. Wait 1 min or enable billing at [Google AI pricing](https://ai.google.dev/pricing) for higher limits.

## Security

- Never commit your API key to version control
- `.env` is typically in `.gitignore` – keep it that way
- Rotate keys periodically
