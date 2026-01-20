"""
Application configuration management.

Centralized configuration using Pydantic Settings with environment variable support.
Validates all configuration parameters at startup to fail fast on misconfiguration.
"""

import logging
import sys
import os
from enum import Enum

from typing import Any, List, Optional, Union, cast

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be overridden via environment variables or .env file.
    Required settings will raise ValidationError if not provided.
    """

    # Application Configuration
    PROJECT_NAME: str = Field(
        default="Black-Scholes Advanced Option Pricing Platform", description="Application name"
    )
    ENVIRONMENT: str = Field(default="dev", description="Environment (dev, staging, prod)")
    DEBUG: bool = Field(default=True, description="Enable debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # Database Configuration
    DATABASE_URL: str = Field(
        default="postgresql://admin@postgres:5432/bsopt",
        description="PostgreSQL database connection string",
    )

    # Redis Configuration
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string for caching and session management",
    )
    REDIS_HOST: str = Field(default="localhost", description="Redis host")
    REDIS_PORT: int = Field(default=6379, description="Redis port")
    REDIS_DB: int = Field(default=0, description="Redis database index")
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Redis password")

    # RabbitMQ Configuration
    RABBITMQ_URL: str = Field(
        default="amqp://admin@localhost:5672/bsopt",
        description="RabbitMQ connection string for async task processing",
    )

    # Rate Limiting Tiers
    RATE_LIMIT_FREE: int = 100
    RATE_LIMIT_PRO: int = 1000
    RATE_LIMIT_ENTERPRISE: int = 10000

    # JWT Authentication
    JWT_SECRET: str = Field(default="change-me-in-production", description="Secret key for secondary JWT/HMAC operations")
    JWT_ALGORITHM: str = Field(default="RS256", description="JWT signing algorithm")
    JWT_PRIVATE_KEY_PATH: str = Field(
        default="certs/jwt-private.pem", description="Path to the JWT private key file"
    )
    JWT_PUBLIC_KEY_PATH: str = Field(
        default="certs/jwt-public.pem", description="Path to the JWT public key file"
    )
    JWT_PRIVATE_KEY: str = ""
    JWT_PUBLIC_KEY: str = ""
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30, description="JWT access token expiration time in minutes"
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7, description="JWT refresh token expiration time in days"
    )
    PASSWORD_MIN_LENGTH: int = Field(default=8, description="Minimum password length requirement")
    PASSWORD_REQUIRE_UPPERCASE: bool = Field(
        default=True, description="Require at least one uppercase letter in password"
    )
    PASSWORD_REQUIRE_LOWERCASE: bool = Field(
        default=True, description="Require at least one lowercase letter in password"
    )
    PASSWORD_REQUIRE_DIGIT: bool = Field(
        default=True, description="Require at least one digit in password"
    )
    PASSWORD_REQUIRE_SPECIAL: bool = Field(
        default=False, description="Require at least one special character in password"
    )
    BCRYPT_ROUNDS: int = Field(
        default=12, description="Number of bcrypt hashing rounds (higher = more secure but slower)"
    )

    # ML Serving Configuration
    ML_SERVICE_URL: str = Field(
        default="http://localhost:8080",
        description="URL of the machine learning serving microservice",
    )
    ML_SERVICE_HOST: str = Field(default="127.0.0.1", description="Host to bind ML serving to")
    ML_SERVICE_PORT: int = Field(default=8080, description="Port to bind ML serving to")

    # CORS Configuration
    CORS_ORIGINS: Union[str, List[str]] = Field(
        default=["http://localhost:3000", "http://localhost:80"], description="Allowed CORS origins"
    )

    # Breach Notification Configuration
    DPA_EMAIL: str = Field(
        default="your-dpa@example.com", description="Email address of the Data Protection Authority"
    )

    # Database Configuration
    SLOW_QUERY_THRESHOLD_MS: int = Field(
        default=100, description="Threshold in milliseconds for logging slow database queries"
    )

    # ML Training Configuration
    ML_TRAINING_TEST_SIZE: float = Field(default=0.2, description="Proportion of dataset to include in the test split for ML training")
    ML_TRAINING_RANDOM_STATE: int = Field(default=42, description="Random state for reproducibility in ML training")
    ML_TRAINING_KFOLD_SPLITS: int = Field(default=3, description="Number of folds for K-fold cross-validation in HPO")
    ML_TRAINING_OPTUNA_TRIALS: int = Field(default=50, description="Number of trials for Optuna hyperparameter optimization")
    ML_TRAINING_PROMOTE_THRESHOLD_R2: float = Field(default=0.98, description="R2 score threshold for promoting ML models to production")
    ML_TRAINING_NN_EPOCHS: int = Field(default=10, description="Number of epochs for Neural Network training")
    ML_TRAINING_NN_LR: float = Field(default=0.001, description="Learning rate for Neural Network training")

    # Dask Configuration for Distributed ML Training
    DASK_LOCAL_CLUSTER_THREADS_PER_WORKER: int = Field(default=2, description="Number of threads per worker in local Dask cluster")
    DASK_ARRAY_DEFAULT_CHUNKS_FRACTION: int = Field(default=4, description="Fraction of data length to use for Dask array chunk size (e.g., 4 means len(X)//4)")


    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is one of the allowed values."""
        v_lower = v.lower()
        if v_lower == "production":
            v_lower = "prod"
        if v_lower == "test":
            v_lower = "dev"
            
        allowed = ["dev", "staging", "prod"]
        if v_lower not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of {allowed}")
        return v_lower

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid logging level."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return v.upper()

    @field_validator("ACCESS_TOKEN_EXPIRE_MINUTES")
    @classmethod
    def validate_token_expiration(cls, v: int) -> int:
        """Validate token expiration is positive."""
        if v <= 0:
            raise ValueError("ACCESS_TOKEN_EXPIRE_MINUTES must be positive")
        return v

    @field_validator("REFRESH_TOKEN_EXPIRE_DAYS")
    @classmethod
    def validate_refresh_expiration(cls, v: int) -> int:
        """Validate refresh token expiration is positive."""
        if v <= 0:
            raise ValueError("REFRESH_TOKEN_EXPIRE_DAYS must be positive")
        return v

    @field_validator("BCRYPT_ROUNDS")
    @classmethod
    def validate_bcrypt_rounds(cls, v: int) -> int:
        """Validate bcrypt rounds are within safe range."""
        if v < 10 or v > 14:
            raise ValueError("BCRYPT_ROUNDS must be between 10 and 14 for security and performance")
        return v

    @field_validator("PASSWORD_MIN_LENGTH")
    @classmethod
    def validate_password_min_length(cls, v: int) -> int:
        """Validate password minimum length."""
        if v < 8:
            raise ValueError("PASSWORD_MIN_LENGTH must be at least 8")
        return v

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> List[str]:
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            # Split by comma, strip whitespace, and filter out empty strings
            origins = [origin.strip() for origin in v.split(",") if origin.strip()]
            return origins
        elif isinstance(v, list):
            # If it's already a list, ensure all elements are strings and stripped
            return [str(item).strip() for item in v if str(item).strip()]
        # If input is invalid or empty, return an empty list to avoid errors
        return []

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "prod"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == "dev"

    @property
    def rate_limit_tiers(self) -> dict:
        """Get rate limit configuration for all tiers."""
        return {
            "free": self.RATE_LIMIT_FREE,
            "pro": self.RATE_LIMIT_PRO,
            "enterprise": self.RATE_LIMIT_ENTERPRISE,
        }


Settings.model_rebuild()


def get_settings() -> Settings:
    """
    Get application settings singleton.

    This function caches the settings instance and loads key files.

    Returns:
        Settings: Validated application settings

    Raises:
        ValidationError: If configuration is invalid
        FileNotFoundError: If key files are not found
    """
    # Use a simple global cache for settings
    global _settings
    if _settings:
        return _settings

    try:
        settings_obj = Settings()

        # Load JWT keys from files
        try:
            with open(settings_obj.JWT_PRIVATE_KEY_PATH, "r") as f:
                settings_obj.JWT_PRIVATE_KEY = f.read()
            with open(settings_obj.JWT_PUBLIC_KEY_PATH, "r") as f:
                settings_obj.JWT_PUBLIC_KEY = f.read()
        except FileNotFoundError as e:
            logger.error(f"JWT key file not found: {e}. Ensure keys are generated.")
            raise

        _settings = settings_obj
        return settings_obj
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def configure_logging(settings: Settings) -> None:
    """
    Configure application logging based on settings.

    Args:
        settings: Application settings containing log level configuration
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - " "%(funcName)s:%(lineno)d - %(message)s"

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
        ],
    )

    # Set third-party library log levels
    if not settings.DEBUG:
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        logging.getLogger("fastapi").setLevel(logging.WARNING)


# Global settings instance
_settings: Optional[Settings] = None
settings: Settings = cast(Settings, None)  # Exported name

# Only initialize settings and exit if not running under pytest
if "pytest" not in sys.modules and not os.environ.get("BSOPT_TEST_MODE"): # pragma: no cover
    try:
        _settings = get_settings()
        settings = _settings
        configure_logging(_settings)
    except (ValidationError, FileNotFoundError) as e:
        # Handle critical configuration errors at import time
        logging.error(f"Failed to load or configure settings: {e}")
        sys.exit(1)
