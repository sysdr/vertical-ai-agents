from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    gemini_api_key: str = "AIzaSyDGswqDT4wQw_bd4WZtIgYAawRDZ0Gisn8"
    host: str = "0.0.0.0"
    port: int = 8000
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    max_seq_length: int = 512
    
    class Config:
        env_file = ".env"

settings = Settings()
