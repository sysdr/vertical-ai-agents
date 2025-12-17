from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    gemini_api_key: str = "mock-api-key"
    gemini_model: str = "gemini-1.5-flash"
    use_mock_gemini: bool = True
    
    # Rate Limiting
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 100000
    
    # Circuit Breaker
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    
    # Cost Configuration
    cost_input_token: float = 0.075  # per 1M tokens
    cost_output_token: float = 0.300  # per 1M tokens
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_enabled: bool = False
    
    # Server Configuration
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
