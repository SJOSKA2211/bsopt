"""

Application configuration management.
"""

import base64
import os

import structlog
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)

# Loaded from environment; never hardcode the actual value in source.
_DEFAULT_DEV_MFA_KEY = os.environ.get("_DEFAULT_DEV_MFA_KEY", "INSECURE_DEV_PLACEHOLDER")

_PRODUCTION_ENVIRONMENTS = {"prod", "production"}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application Configuration
    PROJECT_NAME: str = "BSOpt"
    ENVIRONMENT: str = Field(default="dev")
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Database Configuration
    DATABASE_URL: str = Field(validation_alias="DATABASE_URL")
    DATABASE_MIN_POOL_SIZE: int = 2
    DATABASE_MAX_POOL_SIZE: int = 8
    DATABASE_POOL_TIMEOUT: int = 30
    SLOW_QUERY_THRESHOLD_MS: int = 100

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v.startswith("postgresql://") and not v.startswith("postgresql+asyncpg://"):
            if "sqlite" not in v:
                raise ValueError("DATABASE_URL must be a valid PostgreSQL connection string.")
        return v

    # Redis Configuration
    REDIS_URL: str = Field(validation_alias="REDIS_URL")
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str | None = Field(default=None, validation_alias="REDIS_PASSWORD")

    # RabbitMQ Configuration
    RABBITMQ_USER: str = Field(default="guest", validation_alias="RABBITMQ_USER")
    RABBITMQ_PASSWORD: str = Field(default="guest", validation_alias="RABBITMQ_PASSWORD")
    RABBITMQ_HOST: str = Field(default="rabbitmq", validation_alias="RABBITMQ_HOST")

    @property
    def RABBITMQ_URL(self) -> str:
        """Constructs the RabbitMQ URL from credentials."""
        return f"amqp://{self.RABBITMQ_USER}:{self.RABBITMQ_PASSWORD}@{self.RABBITMQ_HOST}:5672//"

    # ML Serving Configuration
    ML_SERVICE_GRPC_URL: str = "worker:50051"

    # Security Configuration
    OPA_URL: str = Field(default="http://opa:8181/v1/data/authz/allow", validation_alias="OPA_URL")

    # Pricing Configuration
    MONTE_CARLO_GPU_THRESHOLD: int = 10000
    PRICING_LARGE_BATCH_THRESHOLD: int = 1000
    MAX_NET_DELTA: float = 10000.0

    # ML Training Configuration
    ML_TRAINING_DEFAULT_SAMPLES: int = 1000
    ML_TRAINING_OPTUNA_TRIALS: int = 50
    ML_TRAINING_RANDOM_STATE: int = 42
    ML_TRAINING_PROMOTE_THRESHOLD_R2: float = 0.95

    # Email Configuration
    SENDGRID_API_KEY: str = "mock_key"
    DEFAULT_FROM_EMAIL: str = "noreply@bsopt.ai"
    DPA_EMAIL: str = "dpa@bsopt.ai"

    # Rate Limiting Tiers
    RATE_LIMIT_FREE: int = 100
    RATE_LIMIT_PRO: int = 1000
    RATE_LIMIT_ENTERPRISE: int = 10000

    @property
    def rate_limit_tiers(self) -> dict[str, int]:
        """Maps user tiers to their rate limits."""
        return {
            "free": self.RATE_LIMIT_FREE,
            "pro": self.RATE_LIMIT_PRO,
            "enterprise": self.RATE_LIMIT_ENTERPRISE,
        }

    # CORS Configuration
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # JWT Authentication
    JWT_SECRET: str = Field(validation_alias="JWT_SECRET")
    JWT_ALGORITHM: str = "RS256"
    JWT_PRIVATE_KEY: str | None = ""
    JWT_PUBLIC_KEY: str | None = ""
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Authentication Policy
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_LOWERCASE: bool = True
    PASSWORD_REQUIRE_DIGIT: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = True
    REQUIRE_EMAIL_VERIFICATION: bool = False
    MFA_ENCRYPTION_KEY: str = Field(
        default=_DEFAULT_DEV_MFA_KEY,
        validation_alias="MFA_ENCRYPTION_KEY",
    )

    # Password Hashing
    BCRYPT_ROUNDS: int = 12
    ARGON2_TIME_COST: int = 3
    ARGON2_MEMORY_COST: int = 65536
    ARGON2_PARALLELISM: int = 4

    # NSE Scraper Configuration
    NSE_CACHE_TTL: int = 300
    NSE_NAME_SYMBOL_MAP: dict[str, str] = {
        "Safaricom": "SCOM",
        "KCB": "KCB",
        "Equity": "EQTY",
        "Co-operative": "COOP",
    }
    NSE_SECTORS: list[str] = [
        "Banking",
        "Commercial",
        "Energy",
        "Insurance",
        "Investment",
        "Manufacturing",
        "Telecommunication",
    ]

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() in _PRODUCTION_ENVIRONMENTS

    @property
    def rsa_private_key(self) -> str:
        """Returns the private key, ensuring it exists."""
        if self.JWT_PRIVATE_KEY:
            return self.JWT_PRIVATE_KEY
        if self.is_production:
            raise ValueError("JWT_PRIVATE_KEY is missing in production")
        return self._get_transient_key("private")

    @property
    def rsa_public_key(self) -> str:
        """Returns the public key, ensuring it exists."""
        if self.JWT_PUBLIC_KEY:
            return self.JWT_PUBLIC_KEY
        if self.is_production:
            raise ValueError("JWT_PUBLIC_KEY is missing in production")
        return self._get_transient_key("public")

    _transient_keys: dict[str, str] = {}

    def _get_transient_key(self, key_type: str) -> str:
        """Generates or retrieves a transient RSA key for development."""
        if not self._transient_keys:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import rsa

            private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            self._transient_keys["private"] = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            ).decode("utf-8")

            self._transient_keys["public"] = (
                private_key.public_key()
                .public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
                .decode("utf-8")
            )

            logger.warning("using_transient_rsa_keys", mode=self.ENVIRONMENT)

        return self._transient_keys[key_type]

    # MLflow tracking URI
    @property
    def tracking_uri(self) -> str:
        """Point MLflow to Postgres always."""
        return self.DATABASE_URL.replace("postgresql+asyncpg", "postgresql")

    # Dask & Distributed
    DASK_LOCAL_CLUSTER_THREADS_PER_WORKER: int = 1
    DASK_ARRAY_DEFAULT_CHUNKS_FRACTION: int = 10

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.test"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"dev", "staging", "prod", "production", "test"}
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of {sorted(allowed)}")
        return v_lower

    @model_validator(mode="after")
    def validate_security_configs(self) -> "Settings":
        """Validate security-critical settings and apply environment defaults."""

        # Enforce email verification in production by default if not set
        # Note: If it's already set in .env or environment, Pydantic preserves it.
        if self.is_production and self.ENVIRONMENT != "test":
            # We don't want to silently change a False to True if the user
            # explicitly set it to False in prod, but we should at least warn.
            # However, usually we want a safe default.
            pass

        if self.ENVIRONMENT.lower() in _PRODUCTION_ENVIRONMENTS:
            # 1. MFA Key Security
            key = self.MFA_ENCRYPTION_KEY
            if key == _DEFAULT_DEV_MFA_KEY or key == "INSECURE_DEV_PLACEHOLDER":
                raise ValueError(
                    "CRITICAL: MFA_ENCRYPTION_KEY must not use the default "
                    "development key in production. Set a secure key via "
                    "the MFA_ENCRYPTION_KEY environment variable."
                )

            try:
                decoded = base64.urlsafe_b64decode(key + "=" * (-len(key) % 4))
                if len(decoded) < 32:
                    raise ValueError(
                        "MFA_ENCRYPTION_KEY is too short. The decoded key must "
                        "be at least 32 bytes (256 bits)."
                    )
            except Exception as e:
                if "MFA_ENCRYPTION_KEY" in str(e):
                    raise
                raise ValueError(
                    "MFA_ENCRYPTION_KEY must be a valid base64url-encoded "
                    f"string of at least 32 bytes: {e}"
                ) from e

            # 2. Email Verification Security
            if not self.REQUIRE_EMAIL_VERIFICATION:
                logger.warning(
                    "security_warning: email verification is DISABLED in production",
                    environment=self.ENVIRONMENT,
                )

        return self


settings = Settings()


def get_settings():
    """Returns the singleton settings instance."""
    return settings
