# ⚡ Quick Fix: API Issues

## 🔍 Check Your Issue Type

First, check your API status:
```bash
curl http://localhost:8000/api-key-status | python3 -m json.tool
```

Look for:
- `"is_quota_error": true` → You hit rate limits (see **Quota Exceeded** below)
- `"is_valid": false` → Your API key is invalid (see **Invalid API Key** below)

---

## ⏱️ Issue 1: Quota Exceeded (Rate Limit - 429 Error)

### What Happened?
Your API key is **valid** but has hit the **free tier rate limits**. This happens when you make too many requests in a short time.

### ✅ Solutions (Choose One)

#### Option 1: Wait and Retry ⏰
- Some rate limits reset automatically (per-minute / per-hour / per-day depending on metric)
- Check status: `curl http://localhost:8000/api-key-status`

> Important: if your error includes **`limit: 0`** for **`generate_content_free_tier_*`** metrics, waiting won’t help — your **project has 0 free-tier quota** (plan/billing/project setup issue). See Option 0 below.

#### Option 0 (Most Common Fix Here): Your Project Has 0 Free-Tier Quota (limit=0) ✅
If you see `generate_content_free_tier_requests, limit: 0` (or similar), you need to:

- Create a **NEW API key in a NEW project** (AI Studio): `https://makersuite.google.com/app/apikey`
  - Choose “Create API key in new project”
- Or **enable billing / upgrade** for the current project
- Check usage/quota: `https://ai.dev/usage?tab=rate-limit`

#### Option 0c: Keep Working Without LLM (Fallback Chat Mode) ✅
By default, the backend now supports a **fallback planner**: if Gemini is unavailable (quota=0), `POST /chat` will still execute obvious tools heuristically.

- **Disable fallback (optional)**:

```bash
export ALLOW_FALLBACK_PLANNER=false
pkill -f "python main.py"
cd backend && source venv/bin/activate && python main.py
```

#### Option 0b: Model Name Not Supported (404 model not found) ✅
If you see `models/<name> is not found ... or not supported for generateContent`, set a supported model and restart:

```bash
export GEMINI_MODEL=gemini-2.0-flash
pkill -f "python main.py"
cd backend && source venv/bin/activate && python main.py
```

#### Option 2: Get a NEW API Key 🔑 (Recommended)
Create a new API key in a **different Google Cloud project**:

1. Go to: **https://makersuite.google.com/app/apikey**
2. Sign in with Google
3. Click "Create API Key" → Select "Create API key in new project"
4. Copy the new key
5. Run: `./scripts/fix_api_key.sh` and paste the new key

#### Option 3: Upgrade to Paid Tier 💳
- Higher rate limits
- More requests per minute/day
- Visit: https://ai.google.dev/pricing

#### Option 4: Test Without API Key 🧪
Use the direct tool execution endpoint (no LLM needed):

```bash
# Test weather tool
curl -X POST http://localhost:8000/execute-tool \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "get_weather", "inputs": {"city": "Tokyo"}}'

# Test revenue calculator
curl -X POST http://localhost:8000/execute-tool \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "calculate_revenue", "inputs": {"year": 2024, "quarter": 3}}'
```

---

## 🔑 Issue 2: Invalid API Key

### What Happened?
Your API key is expired, invalid, or not configured.

### 🔧 Fix in 30 Seconds

#### Step 1: Get API Key
Visit: **https://makersuite.google.com/app/apikey**
- Sign in with Google
- Click "Create API Key"
- Copy the key (starts with `AIza...`)

#### Step 2: Set It Up
```bash
./scripts/fix_api_key.sh
```
Paste your key when prompted. Done! ✅

#### Step 3: Verify
```bash
curl http://localhost:8000/api-key-status | python3 -m json.tool
```
Look for `"is_valid": true`

---

## 📋 Manual Setup (Alternative)

If you prefer manual setup:

```bash
# 1. Get API key from https://makersuite.google.com/app/apikey

# 2. Create .env file
cd backend
echo "GEMINI_API_KEY=your_actual_api_key_here" > .env

# 3. Restart backend
pkill -f "python main.py"
source venv/bin/activate
python main.py
```

---

## 🧪 Test Your Setup

### Test with LLM (requires valid API key with quota):
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in Tokyo?", "max_turns": 5}'
```

### Test without LLM (no API key needed):
```bash
curl -X POST http://localhost:8000/execute-tool \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "get_weather", "inputs": {"city": "Tokyo"}}'
```

---

## ❓ Need More Help?

- Check status: `curl http://localhost:8000/api-key-status`
- View backend logs: `tail -f backend.log`
- See full solution: `SOLUTION.md`
- API docs: http://localhost:8000/docs

---

## 📊 Understanding Rate Limits

**Free Tier Limits (as of 2024):**
- Requests per minute: ~15
- Requests per day: ~1,500
- These limits reset hourly/daily

**Why you might hit limits:**
- Testing multiple queries quickly
- Running automated tests
- Multiple users sharing same key

**Best practices:**
- Use `/execute-tool` for testing tool logic (bypasses LLM)
- Create separate API keys for different projects
- Implement rate limiting in your application
- Consider paid tier for production use
