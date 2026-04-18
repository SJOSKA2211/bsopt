import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """Application-wide settings managed by Pydantic."""
    
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "dev")
    PROJECT_NAME: str = "BSOPT"
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://admin:password@localhost:5432/bsopt")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Auth
    JWT_SECRET: str = os.getenv("JWT_SECRET", "super-dev-secret-change-me-in-prod")
    JWT_ALGORITHM: str = "HS256"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
