# Rate Limit Information

## Current Issue
You're hitting the **free tier rate limit** for Gemini API.

## Free Tier Limits
- **5 requests per minute** per model
- Applies to: `gemini-2.5-flash`, `gemini-1.5-flash`, etc.

## Solutions

### Option 1: Wait and Retry (Automatic)
The code now automatically:
- Detects rate limit errors (429)
- Extracts the retry delay from the error
- Waits and retries up to 3 times
- Shows a clear error message if all retries fail

**Just wait ~24 seconds and try again!**

### Option 2: Use Different Models
The system will try different models if one hits the limit. You can also:
- Wait between requests
- Use the dashboard less frequently
- Space out your test requests

### Option 3: Upgrade Your Plan
If you need more requests:
1. Visit: https://ai.google.dev/pricing
2. Check available plans
3. Upgrade for higher rate limits

## How It Works Now

The backend now handles rate limits automatically:
1. **Detects 429 errors** (quota exceeded)
2. **Extracts retry delay** from the error message
3. **Waits** for the specified time (e.g., 24 seconds)
4. **Retries** the request automatically
5. **Shows clear message** if still rate-limited after retries

## Best Practices

1. **Space out requests**: Don't click buttons rapidly
2. **Wait between tests**: Give it 10-15 seconds between requests
3. **Use one model**: Stick to one model to avoid hitting multiple limits
4. **Monitor usage**: Check https://ai.dev/usage?tab=rate-limit

## Current Status

✅ **Rate limit handling added**
- Automatic retry with backoff
- Clear error messages
- Extracts retry delay from API response

The system will now automatically handle rate limits and retry when possible!

