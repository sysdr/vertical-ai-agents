"""
LLM client with retry on 429 quota errors and model fallback chain.
"""
import asyncio
import re
import logging
import os
import time
import google.generativeai as genai

logger = logging.getLogger(__name__)

MODEL_FALLBACK_CHAIN = [
    "gemini-2.0-flash-lite",
    "gemini-exp-1206",
    "gemini-2.5-flash",
    "gemini-flash-lite-latest",
    "gemini-pro-latest",
]


def _parse_retry_seconds(err_msg: str) -> int:
    match = re.search(r"retry[_\s]delay[^\d]*(\d+)", err_msg, re.I)
    if match:
        return min(int(match.group(1)), 60)
    match = re.search(r"retry in (\d+)[.\s]?\d*s", err_msg, re.I)
    if match:
        return min(int(match.group(1)) + 2, 60)
    return 30


def _is_quota_error(err: Exception) -> bool:
    err_str = str(err).lower()
    return "429" in str(err) or "quota" in err_str or "rate limit" in err_str


def generate_with_retry(prompt, config=None, max_retries=2):
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)

    models = [os.getenv("GEMINI_MODEL")] if os.getenv("GEMINI_MODEL") else []
    models.extend([m for m in MODEL_FALLBACK_CHAIN if m not in models])

    last_err = None
    for model_name in models:
        model = genai.GenerativeModel(model_name)
        for attempt in range(max_retries):
            try:
                if config is not None:
                    response = model.generate_content(prompt, generation_config=config)
                else:
                    response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                last_err = e
                if _is_quota_error(e):
                    delay = _parse_retry_seconds(str(e))
                    if attempt < max_retries - 1:
                        logger.info(f"429 on {model_name}, retry in {delay}s")
                        time.sleep(delay)
                    else:
                        logger.info(f"429 on {model_name}, trying next model")
                        break
                else:
                    raise
    raise last_err


async def generate_with_retry_async(prompt, config=None, max_retries=2):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: generate_with_retry(prompt, config, max_retries))
