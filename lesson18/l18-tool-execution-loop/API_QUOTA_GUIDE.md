# 📊 API Quota Management Guide

## Understanding the Current Situation

Your system is **fully functional**, but the Google Gemini API key has hit **free tier rate limits** (429 error). This is **NOT** an invalid key - it's just temporarily over quota.

## ✅ What's Working

- ✅ **Backend Server**: Running perfectly
- ✅ **Frontend**: Operational
- ✅ **All 4 Tools**: Fully functional
- ✅ **API Key**: Valid but rate-limited
- ✅ **Direct Tool Execution**: Works without API quota

## ⏱️ What's Affected

- ❌ **LLM-powered chat**: Requires API quota
- ❌ **Natural language queries**: Requires API quota
- ❌ **Tool orchestration via LLM**: Requires API quota

---

## 🎯 Solutions (Choose What Works Best)

### Solution 1: Wait for Quota Reset ⏰ (No Action Required)

**Timeline:** ~1 hour (free tier resets hourly)

**What to do:**
- Nothing! Just wait
- Check status periodically:
  ```bash
  curl http://localhost:8000/api-key-status | python3 -m json.tool
  ```
- When `"is_valid": true`, you're good to go

**Best for:** Non-urgent testing, learning, development

---

### Solution 2: Get New API Key 🔑 (Recommended for Now)

Create a **new API key in a different Google Cloud project**:

#### Step-by-Step:

1. **Visit:** https://makersuite.google.com/app/apikey

2. **Create New Key:**
   - Click "Create API Key"
   - Select **"Create API key in new project"** (important!)
   - Copy the key (starts with `AIza...`)

3. **Install New Key:**
   ```bash
   cd /home/systemdrllp5/git/vertical-ai-agents/lesson18/l18-tool-execution-loop
   ./scripts/fix_api_key.sh
   ```
   Paste your new key when prompted.

4. **Verify:**
   ```bash
   curl http://localhost:8000/api-key-status | python3 -m json.tool
   ```
   Look for: `"is_valid": true`, `"is_quota_error": false`

**Best for:** Immediate continued testing

---

### Solution 3: Test Without LLM 🧪 (Available Now!)

Use the **direct tool execution endpoint** - no API quota needed!

#### Examples:

```bash
# Test Weather Tool
curl -X POST http://localhost:8000/execute-tool \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "get_weather", "inputs": {"city": "Tokyo", "unit": "celsius"}}'

# Test Revenue Calculator
curl -X POST http://localhost:8000/execute-tool \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "calculate_revenue", "inputs": {"year": 2024, "quarter": 3}}'

# Test Product Search
curl -X POST http://localhost:8000/execute-tool \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "search_products", "inputs": {"query": "laptop", "max_results": 3}}'

# Test Document Search
curl -X POST http://localhost:8000/execute-tool \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "document_search", "inputs": {"query": "architecture", "top_k": 2}}'
```

#### View Statistics:
```bash
curl http://localhost:8000/stats | python3 -m json.tool
```

**Best for:** Testing tool implementations, debugging tool logic

---

### Solution 4: Upgrade to Paid Tier 💳 (Production)

**Benefits:**
- Much higher rate limits
- More requests per minute/day
- Better for production use
- No daily/hourly resets

**Pricing:** https://ai.google.dev/pricing

**When to consider:**
- Building production applications
- Need consistent high volume
- Multiple users/sessions

---

## 📈 Understanding Free Tier Limits

### Current Limits (Gemini 2.0 Flash Experimental):

| Metric | Free Tier Limit |
|--------|----------------|
| Requests per minute | ~15 |
| Requests per day | ~1,500 |
| Token input per minute | Limited |

### Why You Hit Limits:

- ✓ Testing multiple queries quickly
- ✓ Multiple validation attempts
- ✓ Automated testing
- ✓ Experimental model has stricter limits

### Rate Limit Reset Schedule:

- **Per-minute limits:** Reset every minute
- **Per-hour limits:** Reset every hour
- **Per-day limits:** Reset at midnight UTC

---

## 🔍 Check Your Status

### Quick Status Check:
```bash
curl http://localhost:8000/api-key-status | python3 -m json.tool
```

### Detailed Health Check:
```bash
curl http://localhost:8000/health | python3 -m json.tool
```

### View Backend Logs:
```bash
tail -f /home/systemdrllp5/git/vertical-ai-agents/lesson18/l18-tool-execution-loop/backend.log
```

---

## 🛠️ Best Practices

### For Development:

1. **Use `/execute-tool` for tool testing** (no API quota)
2. **Test LLM integration sparingly** (uses quota)
3. **Create separate API keys per project**
4. **Monitor your usage:** https://ai.dev/usage

### For Production:

1. **Implement rate limiting** in your app
2. **Cache LLM responses** where possible
3. **Use paid tier** for reliability
4. **Monitor quota usage** proactively
5. **Handle 429 errors gracefully** (retry with backoff)

### Quota-Conscious Testing:

```bash
# ✅ Good: Test tools directly (no quota used)
curl -X POST http://localhost:8000/execute-tool -H "Content-Type: application/json" -d '{"tool_name": "get_weather", "inputs": {"city": "Tokyo"}}'

# ⚠️ Careful: Each LLM call uses quota
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"query": "What is the weather?", "max_turns": 5}'
```

---

## 📚 Additional Resources

- **Google AI Studio:** https://makersuite.google.com/app/apikey
- **API Documentation:** https://ai.google.dev/gemini-api/docs
- **Rate Limits Guide:** https://ai.google.dev/gemini-api/docs/rate-limits
- **Usage Dashboard:** https://ai.dev/usage
- **Pricing Info:** https://ai.google.dev/pricing

---

## 🎓 What You've Learned

This quota situation is actually a **great learning opportunity**:

1. ✅ How to handle API rate limits
2. ✅ Building fallback mechanisms (`/execute-tool`)
3. ✅ Distinguishing between invalid keys vs quota issues
4. ✅ Production-ready error handling
5. ✅ Quota management strategies

Your system is architected well - it continues to function even when the LLM API is rate-limited!

---

## 🚀 Next Steps

**Right Now:**
1. Use `/execute-tool` to continue testing tool implementations
2. OR wait ~30 minutes for quota reset
3. OR get a new API key in a different project

**For Production:**
1. Implement proper rate limiting
2. Add retry logic with exponential backoff
3. Consider paid tier for consistent performance
4. Add monitoring and alerting

---

## ❓ FAQ

**Q: Is my API key broken?**  
A: No! It's valid but temporarily over quota.

**Q: Will this happen in production?**  
A: Only if you stay on free tier. Paid tier has much higher limits.

**Q: Can I test without fixing this?**  
A: Yes! Use `/execute-tool` endpoint for direct tool testing.

**Q: How long until reset?**  
A: Usually ~1 hour for hourly limits, midnight UTC for daily limits.

**Q: Should I delete my current key?**  
A: No need. Just create a new one in a different project.

---

*Last Updated: 2026-01-07*


