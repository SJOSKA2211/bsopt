"""
Application configuration management.
Neon and OAuth 2.0 optimized.
"""

import os
from typing import List, Optional, Union, Dict
import structlog
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application Configuration
    PROJECT_NAME: str = "BSOpt Singularity"
    ENVIRONMENT: str = Field(default="dev")
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Database Configuration (Neon Optimized)
    # Neon uses 'postgresql://' strings. Ensure we handle pooled connections.
    DATABASE_URL: str = Field(validation_alias="DATABASE_URL")
    SLOW_QUERY_THRESHOLD_MS: int = 100

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v.startswith("postgresql://") and not v.startswith("postgresql+asyncpg://"):
            if "sqlite" not in v:
                raise ValueError("DATABASE_URL must be a PostgreSQL connection string for Neon integration.")
        return v

    # Redis Configuration
    REDIS_URL: str = Field(validation_alias="REDIS_URL")
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    # RabbitMQ Configuration
    RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672//"

    # ML Serving Configuration
    ML_SERVICE_GRPC_URL: str = "localhost:50051"
    
    # Rate Limiting Tiers
    RATE_LIMIT_FREE: int = 100
    RATE_LIMIT_PRO: int = 1000
    RATE_LIMIT_ENTERPRISE: int = 10000

    # CORS Configuration
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # JWT Authentication
    JWT_SECRET: str = Field(validation_alias="JWT_SECRET")
    JWT_ALGORITHM: str = "RS256"
    JWT_PRIVATE_KEY: Optional[str] = ""
    JWT_PUBLIC_KEY: Optional[str] = ""
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    @property
    def rsa_private_key(self) -> str:
        """Returns the private key, ensuring it exists."""
        if self.JWT_PRIVATE_KEY:
            return self.JWT_PRIVATE_KEY
        if self.ENVIRONMENT == "prod":
            raise ValueError("JWT_PRIVATE_KEY is missing in production")
        return self._get_transient_key("private")

    @property
    def rsa_public_key(self) -> str:
        """Returns the public key, ensuring it exists."""
        if self.JWT_PUBLIC_KEY:
            return self.JWT_PUBLIC_KEY
        if self.ENVIRONMENT == "prod":
            raise ValueError("JWT_PUBLIC_KEY is missing in production")
        return self._get_transient_key("public")

    _transient_keys: Dict[str, str] = {}

    def _get_transient_key(self, key_type: str) -> str:
        """Generates or retrieves a transient RSA key for development."""
        if not self._transient_keys:
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.hazmat.primitives import serialization
            
            private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            self._transient_keys["private"] = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ).decode("utf-8")
            
            self._transient_keys["public"] = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode("utf-8")
            
            logger.warning("using_transient_rsa_keys", mode=self.ENVIRONMENT)
            
        return self._transient_keys[key_type]

    # MLflow tracking URI
    @property
    def tracking_uri(self) -> str:
        """Point MLflow to Postgres in production."""
        if self.ENVIRONMENT == "prod":
            return self.DATABASE_URL.replace("postgresql+asyncpg", "postgresql")
        return "sqlite:///mlflow.db"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = ["dev", "staging", "prod", "test"]
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of {allowed}")
        return v_lower

settings = Settings()

def get_settings():
    """Returns the singleton settings instance."""
    return settings