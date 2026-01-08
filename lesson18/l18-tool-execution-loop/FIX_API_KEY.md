# 🔧 Fix API Key Issue

## Problem
The API key is not configured or has expired, causing this error:
```
🔑 API Key not configured! Please set up your Google Gemini API key...
```

## ✅ Solution

### Quick Fix (3 commands):

1. **Get your API key** from: https://makersuite.google.com/app/apikey
   - Sign in with Google
   - Click "Create API Key"
   - Copy the key

2. **Set the API key**:
   ```bash
   cd backend
   echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
   ```

3. **Restart the backend**:
   ```bash
   pkill -f "python main.py"
   cd backend
   source venv/bin/activate
   python main.py
   ```

### Or use the automated script:
```bash
./scripts/setup_api_key.sh
```

## ✅ Verify It's Fixed

Test with:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "max_turns": 1}'
```

If you get a response (not an API key error), it's working!

## 📚 More Help

See `API_KEY_SETUP_GUIDE.md` for detailed instructions and troubleshooting.


