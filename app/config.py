from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    groq_api_key: str
    s3_bucket_name: str
    aws_region: str = "eu-north-1"
    model_name: str = "llama-3.1-8b-instant"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()