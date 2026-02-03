from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "ReAct Planning System"
    VERSION: str = "1.0.0"
    GEMINI_API_KEY: str = ""
    MAX_ITERATIONS: int = 10
    REASONING_TIMEOUT: int = 30

    class Config:
        env_file = Path(__file__).resolve().parent.parent.parent / ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
