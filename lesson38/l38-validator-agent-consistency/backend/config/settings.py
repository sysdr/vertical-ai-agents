from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "L38 ValidatorAgent - Consistency"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Gemini AI (set via .env or environment)
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash-exp"
    
    # Redis Cache
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    CACHE_TTL: int = 3600  # 1 hour
    
    # Validation Settings
    MAX_CONCURRENT_VALIDATIONS: int = 5
    VALIDATION_TIMEOUT: int = 10
    CONSISTENCY_THRESHOLD: float = 0.5
    TEMPORAL_DRIFT_DAYS: int = 90
    
    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 30
    
    class Config:
        env_file = ".env"

settings = Settings()
