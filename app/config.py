from pydantic_settings import BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    # API Keys
    groq_api_key: str
    
    # Environment
    environment: str = "development"
    
    # Model settings
    model_name: str = "llama-3.1-8b-instant"
    temperature: float = 0.3
    max_tokens: int = 1024
    
    # RAG settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Local paths
    documents_path: str = "./data/documents"
    chroma_db_path: str = "./chroma_db"
    
    # AWS settings (only used in production) IG
    s3_bucket_name: str = ""
    aws_region: str = "eu-north-1"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings():
    return Settings()