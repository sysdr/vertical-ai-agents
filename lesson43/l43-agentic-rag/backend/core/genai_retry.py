"""
L43: Retry helper for Gemini API 429 (quota/rate limit).
Waits for API-suggested retry_delay (e.g. 60s for free tier) then retries.
"""
from __future__ import annotations
import re
import time


def with_429_retry(fn, max_retries: int = 2):
    """Call fn(); on 429 wait ~65s and retry once. After that caller should use fallback (e.g. demo response)."""
    last_exception = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            msg = str(e)
            is_429 = "429" in msg or "quota" in msg.lower() or "rate" in msg.lower()
            if attempt >= max_retries - 1 or not is_429:
                raise
            # Free tier: wait for quota reset then retry once
            match = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", msg, re.I)
            if match:
                delay = float(match.group(1)) + 2.0
            else:
                match = re.search(r"seconds:\s*(\d+)", msg)
                delay = float(match.group(1)) + 2.0 if match else 65.0
            delay = min(70.0, max(15.0, delay))
            time.sleep(delay)
    raise last_exception
