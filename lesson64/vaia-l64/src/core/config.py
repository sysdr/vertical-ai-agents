"""Settings loaded from environment / ConfigMap / Secret."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "models/gemini-2.0-flash-lite"
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    redis_url: str = "redis://redis:6379/0"
    log_level: str = "INFO"
    environment: str = "production"
