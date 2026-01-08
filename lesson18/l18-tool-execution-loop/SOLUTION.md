# 🔧 Solution: API Key Error

## Current Problem
```
❌ API Key Error: Your Google Gemini API key is invalid or expired.
```

The API key in `backend/.env` is **expired** and needs to be replaced.

## ✅ Quick Fix (Choose One Method)

### Method 1: Interactive Script (Easiest)
```bash
./scripts/setup_api_key.sh
```
This script will:
- Prompt you for your API key
- Validate it
- Restart the backend automatically

### Method 2: Manual Setup
1. **Get your API key:**
   - Go to: https://makersuite.google.com/app/apikey
   - Sign in with Google
   - Click "Create API Key"
   - Copy the key (starts with `AIza...`)

2. **Update the .env file:**
   ```bash
   cd backend
   echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
   ```

3. **Restart backend:**
   ```bash
   pkill -f "python main.py"
   cd backend
   source venv/bin/activate
   python main.py
   ```

### Method 3: Environment Variable
```bash
export GEMINI_API_KEY=your_actual_api_key_here
# Then restart backend
```

## ✅ Verify It's Fixed

**Option 1: Use validation script**
```bash
cd backend
source venv/bin/activate
python ../scripts/validate_api_key.py
```

**Option 2: Test the API**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in Tokyo?", "max_turns": 5}'
```

If you get a proper response (not an API key error), it's working! ✅

## 📝 What I've Set Up For You

✅ `.env` file structure is ready  
✅ Validation script: `scripts/validate_api_key.py`  
✅ Setup script: `scripts/setup_api_key.sh` (improved)  
✅ Start script: `scripts/start.sh` (now validates API key)  
✅ Documentation: `API_KEY_SETUP_GUIDE.md` and `FIX_API_KEY.md`

## 🎯 Next Steps

1. Get your API key from Google (link above)
2. Run `./scripts/setup_api_key.sh` and paste your key
3. The system will automatically validate and restart

That's it! The system is fully configured - you just need a valid API key.


