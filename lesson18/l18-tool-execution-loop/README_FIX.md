# 🔧 How to Fix the API Key Error

## The Problem
You're seeing this error:
```
🔑 API Key Error: Your Google Gemini API key is invalid or expired.
```

## ✅ The Solution (Choose One)

### Option 1: Interactive Fix Script (Easiest - Recommended)
```bash
./scripts/fix_api_key.sh
```

This script will:
- ✅ Check your current API key status
- ✅ Guide you to get a new API key
- ✅ Validate the key automatically
- ✅ Restart the backend for you
- ✅ Test that everything works

**Just run it and follow the prompts!**

### Option 2: Quick Setup Script
```bash
./scripts/setup_api_key.sh
```

### Option 3: Manual Fix
1. Get API key from: https://makersuite.google.com/app/apikey
2. Create `.env` file:
   ```bash
   cd backend
   echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
   ```
3. Restart backend:
   ```bash
   pkill -f "python main.py"
   cd backend
   source venv/bin/activate
   python main.py
   ```

## ✅ Verify It's Fixed

### Check API Key Status
```bash
curl http://localhost:8000/api-key-status | python3 -m json.tool
```
Look for `"is_valid": true`

### Test the API
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in Tokyo?", "max_turns": 5}'
```

If you get a proper response (not an error), it's working! ✅

## 📋 What Changed

The system now:
- ✅ Validates API keys on startup
- ✅ Provides clearer error messages
- ✅ Points to the fix script automatically
- ✅ Has a status endpoint to check API key health
- ✅ Better error handling throughout

## 🆘 Still Having Issues?

1. Check the backend logs: `tail -f backend.log`
2. Verify the .env file exists: `cat backend/.env`
3. Check API key status: `curl http://localhost:8000/api-key-status`
4. Run validation: `cd backend && source venv/bin/activate && python ../scripts/validate_api_key.py`

## 📚 More Help

- Quick fix guide: `QUICK_FIX.md`
- Full solution: `SOLUTION.md`
- Setup guide: `API_KEY_SETUP_GUIDE.md`


