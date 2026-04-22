from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Registry:
    """Dynamic component registry (Phase 2)."""
    _components: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, component: Any):
        cls._components[name] = component

    @classmethod
    def get(cls, name: str) -> Any:
        return cls._components.get(name)

class Settings(BaseSettings):
    """
    Application-wide settings.
    Enforced by OMARCHY ABSOLUTE Protocol.
    """
    ENVIRONMENT: str = Field(default="production")
    DEBUG: bool = Field(default=False)
    
    # Database
    DATABASE_URL: str = Field(...)
    REDIS_URL: str = Field(...)
    
    # Auth
    JWT_SECRET: str = Field(...)
    JWT_ALGORITHM: str = "HS256"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
config_registry = Registry()
