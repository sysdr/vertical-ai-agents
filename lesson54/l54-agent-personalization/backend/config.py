import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DB_PATH = os.getenv("DB_PATH", "data/profiles.db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
PRIVACY_MODE = os.getenv("PRIVACY_MODE", "true").lower() == "true"
