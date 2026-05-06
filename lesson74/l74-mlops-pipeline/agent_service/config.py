from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = "gemini-2.0-flash"
    port: int = Field(default=8074, alias="PORT")
    db_path: str = "data/deployment.db"
    drift_threshold: float = 0.35
    error_rate_threshold: float = 0.05
    p95_latency_threshold: float = 3.0
    check_interval_seconds: float = 10.0
    alert_webhook_url: str = ""
    warmup_delay_seconds: float = 2.0
    active_variant: str = "stable"

    model_config = {"env_file": ".env", "populate_by_name": True}


settings = Settings()
