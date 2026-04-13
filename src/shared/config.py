import os
from typing import Annotated, Any

import structlog
from pydantic import AliasChoices, BeforeValidator, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)

_PRODUCTION_ENVIRONMENTS = {"prod", "production"}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application Configuration
    PROJECT_NAME: str = "BS-OPT Unified Manifold"
    ENVIRONMENT: str = Field(default="dev")
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    DEFAULT_TICKER: str = "SPY"

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return v.upper()

    DATABASE_URL: str = Field(
        ...,
        validation_alias="DATABASE_URL",
    )
    DATABASE_MIN_POOL_SIZE: int = 5
    DATABASE_MAX_POOL_SIZE: int = 15
    DATABASE_POOL_TIMEOUT: int = 60
    DATABASE_POOL_RECYCLE: int = 1800
    DATABASE_POOL_PRE_PING: bool = True
    SLOW_QUERY_THRESHOLD_MS: int = 100
    PGBOUNCER_ENABLED: bool = Field(default=True, validation_alias="PGBOUNCER_ENABLED")
    PGBOUNCER_ADMIN_USER: str = Field(..., validation_alias="PGBOUNCER_ADMIN_USER")
    PGBOUNCER_ADMIN_PASSWORD: str = Field(..., validation_alias="PGBOUNCER_ADMIN_PASSWORD")
    PGBOUNCER_HOST: str = Field(..., validation_alias="PGBOUNCER_HOST")
    PGBOUNCER_PORT: int = Field(default=6432, validation_alias="PGBOUNCER_PORT")

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not any(
            v.startswith(prefix)
            for prefix in ["postgresql://", "postgresql+asyncpg://", "postgresql+psycopg://"]
        ):
            if "sqlite" not in v:
                raise ValueError("DATABASE_URL must be a valid PostgreSQL connection string.")
        return v

    # Redis Configuration
    REDIS_HOST: str = Field(..., validation_alias="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, validation_alias="REDIS_PORT")
    REDIS_DB: int = Field(default=0, validation_alias="REDIS_DB")
    REDIS_PASSWORD: str = Field(..., validation_alias="REDIS_PASSWORD")

    @property
    def REDIS_URL(self) -> str:
        """Constructs the Redis URL with auth."""
        return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # RabbitMQ Configuration
    RABBITMQ_USER: str = Field(..., validation_alias="RABBITMQ_USER")
    RABBITMQ_PASSWORD: str = Field(..., validation_alias="RABBITMQ_PASSWORD")
    RABBITMQ_HOST: str = Field(..., validation_alias="RABBITMQ_HOST")
    RABBITMQ_PORT: int = Field(default=5672, validation_alias="RABBITMQ_PORT")

    @property
    def RABBITMQ_URL(self) -> str:
        """Constructs the RabbitMQ URL from credentials."""
        return f"amqp://{self.RABBITMQ_USER}:{self.RABBITMQ_PASSWORD}@{self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}//"

    @property
    def CELERY_BROKER_URL(self) -> str:
        """Constructs the Celery Broker URL from RabbitMQ credentials."""
        return self.RABBITMQ_URL

    # ML Serving Configuration
    ML_SERVICE_GRPC_URL: str = Field(..., validation_alias="ML_SERVICE_GRPC_URL")
    AUTH_SERVICE_GRPC_URL: str = Field(..., validation_alias="AUTH_SERVICE_GRPC_URL")
    NN_MODEL_PATH: str = "models/latest_nn_pricing.onnx"
    XGB_ONNX_MODEL_PATH: str = "models/latest_xgb_pricing.onnx"
    XGB_INT8_MODEL_PATH: str = "models/latest_xgb_pricing.int8.onnx"

    # Security Configuration
    BSOPT_ALLOW_WEAK_SECRETS: bool = Field(
        default=False, validation_alias="BSOPT_ALLOW_WEAK_SECRETS"
    )
    OPA_URL: str = Field(..., validation_alias="OPA_URL")
    AUDIT_VAULT_KEY: str = Field(
        ..., validation_alias="AUDIT_VAULT_KEY"
    )


    @field_validator(
        "AUDIT_VAULT_KEY",
        "RABBITMQ_PASSWORD",
        "REDIS_PASSWORD",
        "BETTER_AUTH_SECRET",
        "JWT_SECRET",
        "PGBOUNCER_ADMIN_PASSWORD",
        "MINIO_ROOT_PASSWORD",
    )
    @classmethod
    def validate_secret_strength(cls, v: str | None, info: Any) -> str | None:
        if v is None:
            return v
        # We need to check if the allow flag is set. 
        # Since this is a field validator, we can't easily check other fields 
        # unless they are already validated. 
        # However, pydantic-settings loads values into os.environ in some cases, 
        # but let's check the env directly or just check os.environ as a fallback.
        allow_weak = os.environ.get("BSOPT_ALLOW_WEAK_SECRETS", "").lower() in ("true", "1", "yes")
        if len(v) < 32 and not allow_weak:
            raise ValueError(
                f"{info.field_name} must be at least 32 characters for production security."
            )
        return v

    # IBM Quantum Configuration
    IBM_QUANTUM_TOKEN: str | None = Field(default=None, validation_alias="IBM_QUANTUM_TOKEN")

    # Pricing Configuration
    PRICING_LARGE_BATCH_THRESHOLD: int = 1000
    MAX_NET_DELTA: float = 10000.0
    MAX_NET_GAMMA: float = 5000.0
    MAX_NET_VEGA: float = 5000.0

    # Market Data Providers
    POLYGON_API_KEY: str | None = Field(default=None, validation_alias="POLYGON_API_KEY")
    ALPHA_VANTAGE_API_KEY: str | None = Field(
        default=None, validation_alias="ALPHA_VANTAGE_API_KEY"
    )

    # Trading Broker Integration
    BROKER_TYPE: str = Field(default="alpaca", validation_alias="BROKER_TYPE")
    BROKER_USE_PAPER: bool = Field(default=True, validation_alias="BROKER_USE_PAPER")
    ALPACA_API_KEY: str = Field(default="", validation_alias="ALPACA_API_KEY")
    ALPACA_API_SECRET: str = Field(default="", validation_alias="ALPACA_API_SECRET")

    # ML Training Configuration
    ML_TRAINING_DEFAULT_SAMPLES: int = 1000
    ML_TRAINING_OPTUNA_TRIALS: int = 50
    ML_TRAINING_RANDOM_STATE: int = 42
    ML_TRAINING_PROMOTE_THRESHOLD_R2: float = Field(
        default=0.95, validation_alias="ML_TRAINING_PROMOTE_THRESHOLD_R2"
    )

    # Email Configuration
    EMAIL_SERVICE_API_KEY: str | None = Field(
        default=None, validation_alias="EMAIL_SERVICE_API_KEY"
    )
    SENDGRID_API_KEY: str | None = Field(default=None, validation_alias="SENDGRID_API_KEY")
    DEFAULT_FROM_EMAIL: str = "noreply@bsopt.ai"
    DPA_EMAIL: str = "dpa@bsopt.ai"

    # ML & Orchestration
    RAY_ADDRESS: str | None = Field(default=None)
    RAY_NAMESPACE: str = "bsopt"
    RAY_SHUTDOWN_AFTER_RUN: bool = Field(default=False)
    MLFLOW_TRACKING_URI: str | None = Field(default=None, validation_alias="MLFLOW_TRACKING_URI")

    @property
    def tracking_uri(self) -> str | None:
        return self.MLFLOW_TRACKING_URI

    # Scrapers & Market Data
    USE_SHM: bool = Field(default=False)

    # Observability & Tracing
    ENABLE_TRACING: bool = False
    PROMETHEUS_URL: str = Field(default="http://prometheus:9090", validation_alias="PROMETHEUS_URL")
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://tempo:4317"
    PUSHGATEWAY_URL: str | None = Field(default=None)
    GRAFANA_URL: str | None = Field(default=None)
    CHAOS_MODE: bool = Field(default=False, validation_alias="BSOPT_CHAOS_MODE")
    LOG_SAMPLING_RATE: float = Field(default=0.1, validation_alias="LOG_SAMPLING_RATE")

    # Rate Limiting Tiers
    RATE_LIMIT_FREE: int = 100
    RATE_LIMIT_PRO: int = 1000
    RATE_LIMIT_ENTERPRISE: int = 10000

    # Trusted Proxies for Zero Trust
    TRUSTED_PROXIES: set[str] = {"127.0.0.1", "::1", "172.16.0.0/12", "10.0.0.0/8"}

    # Market Configuration
    MARKET_TICKER_SYMBOLS: list[str] = Field(
        default=[
            "NIFTY",
            "BANKNIFTY",
            "SPX",
            "BTC-USD",
            "RELIANCE.NR",
            "HDFCBANK.NR",
        ],
        validation_alias="MARKET_TICKER_SYMBOLS",
    )
    DEFAULT_VOLATILITY: float = 0.20
    RISK_FREE_RATE: float = 0.05
    DIVIDEND_YIELD: float = 0.01

    @property
    def rate_limit_tiers(self) -> dict[str, int]:
        """Maps user tiers to their rate limits."""
        return {
            "free": self.RATE_LIMIT_FREE,
            "pro": self.RATE_LIMIT_PRO,
            "enterprise": self.RATE_LIMIT_ENTERPRISE,
        }

    # CORS Configuration
    CORS_ORIGINS: Annotated[
        list[str],
        BeforeValidator(
            lambda v: [
                x.strip()
                for x in (v if isinstance(v, list) else ([v] if isinstance(v, str) else []))
                if x.strip()
            ]
        ),
    ] = ["http://localhost:3000", "http://localhost:5173"]

    # MinIO Configuration
    MINIO_ENDPOINT: str = Field(..., validation_alias="MINIO_ENDPOINT")
    MINIO_ROOT_USER: str = Field(..., validation_alias="MINIO_ROOT_USER")
    MINIO_ROOT_PASSWORD: str = Field(..., validation_alias="MINIO_ROOT_PASSWORD")
    MINIO_USE_SSL: bool = Field(default=False, validation_alias="MINIO_USE_SSL")

    @property
    def MINIO_ENDPOINT_URL(self) -> str:
        """Constructs the MinIO endpoint URL."""
        protocol = "https" if self.MINIO_USE_SSL else "http"
        return f"{protocol}://{self.MINIO_ENDPOINT}"

    # JWT Authentication
    JWT_SECRET: str = Field(..., validation_alias="JWT_SECRET")
    JWT_ALGORITHM: str = "RS256"
    JWT_PRIVATE_KEY: str | None = Field(
        ..., validation_alias=AliasChoices("JWT_PRIVATE_KEY", "JWT_RS256_PRIVATE")
    )
    JWT_PUBLIC_KEY: str | None = Field(
        ..., validation_alias=AliasChoices("JWT_PUBLIC_KEY", "JWT_RS256_PUBLIC")
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    @field_validator("ACCESS_TOKEN_EXPIRE_MINUTES", "REFRESH_TOKEN_EXPIRE_DAYS")
    @classmethod
    def validate_token_expiration(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Token expiration must be a positive integer.")
        return v

    # Authentication Policy
    PASSWORD_MIN_LENGTH: int = 8

    @field_validator("PASSWORD_MIN_LENGTH")
    @classmethod
    def validate_password_min_length(cls, v: int) -> int:
        if v < 8:
            raise ValueError("PASSWORD_MIN_LENGTH must be at least 8.")
        return v

    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_LOWERCASE: bool = True
    PASSWORD_REQUIRE_DIGIT: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = True
    REQUIRE_EMAIL_VERIFICATION: bool = False
    MFA_ENCRYPTION_KEY: str = Field(
        ...,
        validation_alias="MFA_ENCRYPTION_KEY",
    )

    # E2E & Testing
    ALLOW_E2E_EMAIL_BYPASS: bool = Field(default=False, validation_alias="ALLOW_E2E_EMAIL_BYPASS")

    # Better Auth Configuration
    BETTER_AUTH_SECRET: str = Field(..., validation_alias="BETTER_AUTH_SECRET")
    BETTER_AUTH_URL: str = Field(
        ..., validation_alias="BETTER_AUTH_URL"
    )

    # WebAuthn Configuration
    WEBAUTHN_RP_ID: str = Field(default="localhost", validation_alias="WEBAUTHN_RP_ID")
    WEBAUTHN_RP_NAME: str = Field(default="Manifold Auth", validation_alias="WEBAUTHN_RP_NAME")
    WEBAUTHN_ORIGIN: str = Field(
        default="http://localhost:3000", validation_alias="WEBAUTHN_ORIGIN"
    )

    # Social OAuth2 Configuration
    GOOGLE_CLIENT_ID: str | None = Field(default=None, validation_alias="GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET: str | None = Field(default=None, validation_alias="GOOGLE_CLIENT_SECRET")
    GITHUB_CLIENT_ID: str | None = Field(default=None, validation_alias="GITHUB_CLIENT_ID")
    GITHUB_CLIENT_SECRET: str | None = Field(default=None, validation_alias="GITHUB_CLIENT_SECRET")

    # Password Hashing
    BCRYPT_ROUNDS: int = 12

    @field_validator("BCRYPT_ROUNDS")
    @classmethod
    def validate_bcrypt_rounds(cls, v: int) -> int:
        if not (4 <= v <= 15):
            raise ValueError("BCRYPT_ROUNDS must be between 4 and 15.")
        return v

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
        """Returns the private key, ensuring it exists. Decodes from base64 if needed."""
        raw_key = self.JWT_PRIVATE_KEY
        if raw_key:
            import base64

            try:
                # If it's PEM format (starts with -----), return as is
                if raw_key.strip().startswith("-----BEGIN"):
                    return raw_key
                # Otherwise assume it's base64 encoded
                return base64.b64decode(raw_key).decode("utf-8")
            except Exception as e:
                logger.error("failed_to_decode_jwt_private_key", error=str(e))
                if self.is_production:
                    raise
        if self.is_production:
            raise ValueError("JWT_PRIVATE_KEY is missing in production")
        return self._get_transient_key("private")

    @property
    def rsa_public_key(self) -> str:
        """Returns the public key, ensuring it exists. Decodes from base64 if needed."""
        raw_key = self.JWT_PUBLIC_KEY
        if raw_key:
            import base64

            try:
                if raw_key.strip().startswith("-----BEGIN"):
                    return raw_key
                return base64.b64decode(raw_key).decode("utf-8")
            except Exception as e:
                logger.error("failed_to_decode_jwt_public_key", error=str(e))
                if self.is_production:
                    raise
        if self.is_production:
            raise ValueError("JWT_PUBLIC_KEY is missing in production")
        return self._get_transient_key("public")

    # ES256 Keys
    JWT_ES256_PRIVATE: str | None = Field(default=None, validation_alias="JWT_ES256_PRIVATE")
    JWT_ES256_PUBLIC: str | None = Field(default=None, validation_alias="JWT_ES256_PUBLIC")

    @property
    def es256_private_key(self) -> str:
        """Returns the ES256 private key, ensuring it exists. Decodes from base64 if needed."""
        raw_key = self.JWT_ES256_PRIVATE
        if raw_key:
            import base64

            try:
                if raw_key.strip().startswith("-----BEGIN"):
                    return raw_key
                return base64.b64decode(raw_key).decode("utf-8")
            except Exception as e:
                logger.error("failed_to_decode_jwt_es256_private_key", error=str(e))
                if self.is_production:
                    raise
        if self.is_production:
            raise ValueError("JWT_ES256_PRIVATE is missing in production")
        return self._get_transient_key("private_ecc")

    @property
    def es256_public_key(self) -> str:
        """Returns the ES256 public key, ensuring it exists. Decodes from base64 if needed."""
        raw_key = self.JWT_ES256_PUBLIC
        if raw_key:
            import base64

            try:
                if raw_key.strip().startswith("-----BEGIN"):
                    return raw_key
                return base64.b64decode(raw_key).decode("utf-8")
            except Exception as e:
                logger.error("failed_to_decode_jwt_es256_public_key", error=str(e))
                if self.is_production:
                    raise
        if self.is_production:
            raise ValueError("JWT_ES256_PUBLIC is missing in production")
        return self._get_transient_key("public_ecc")

    _transient_keys: dict[str, str] = {}

    def _get_transient_key(self, key_type: str) -> str:
        """Generates or retrieves a transient RSA or ECC key for development."""
        if not self._transient_keys:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import ec, rsa

            # RSA 2048
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

            # ECC P-256
            private_key_ecc = ec.generate_private_key(ec.SECP256R1())
            self._transient_keys["private_ecc"] = private_key_ecc.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            ).decode("utf-8")

            self._transient_keys["public_ecc"] = (
                private_key_ecc.public_key()
                .public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
                .decode("utf-8")
            )

            logger.warning("using_transient_cryptographic_keys", mode=self.ENVIRONMENT)

        return self._transient_keys[key_type]

    # MLflow tracking URI
    @property
    def tracking_uri(self) -> str:
        """Returns the MLflow tracking URI, defaulting to database backend if not set."""
        if self.MLFLOW_TRACKING_URI:
            return self.MLFLOW_TRACKING_URI
        return self.DATABASE_URL.replace("postgresql+asyncpg", "postgresql")

    # Dask & Distributed
    DASK_LOCAL_CLUSTER_THREADS_PER_WORKER: int = 4
    DASK_ARRAY_DEFAULT_CHUNKS_FRACTION: int = 20
    RAY_CPU_PER_NODE: int = 4
    RAY_MEMORY_GB_PER_NODE: int = 4

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.test"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"dev", "development", "staging", "prod", "production", "test"}
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of {sorted(allowed)}")
        return "dev" if v_lower == "development" else v_lower

    @model_validator(mode="after")
    def validate_security_configs(self) -> "Settings":
        """Validate security-critical settings."""

        if self.is_production and not self.BSOPT_ALLOW_WEAK_SECRETS:
            # 1. BETTER_AUTH_SECRET
            if not self.BETTER_AUTH_SECRET or len(self.BETTER_AUTH_SECRET) < 32:
                raise ValueError("CRITICAL: BETTER_AUTH_SECRET must be at least 32 characters in production.")

            # 2. JWT_SECRET
            if not self.JWT_SECRET or len(self.JWT_SECRET) < 32:
                raise ValueError("CRITICAL: JWT_SECRET must be at least 32 characters in production.")

            # 3. MFA_ENCRYPTION_KEY
            if not self.MFA_ENCRYPTION_KEY:
                raise ValueError("CRITICAL: MFA_ENCRYPTION_KEY must be set in production.")

            try:
                import base64
                decoded = base64.urlsafe_b64decode(self.MFA_ENCRYPTION_KEY + "=" * (-len(self.MFA_ENCRYPTION_KEY) % 4))
                if len(decoded) < 32:
                    raise ValueError("MFA_ENCRYPTION_KEY must be at least 32 bytes after base64 decoding.")
            except Exception as e:
                raise ValueError(f"MFA_ENCRYPTION_KEY must be valid base64: {e}")

        return self


settings = Settings()


def get_settings() -> Settings:
    """Returns the singleton settings instance."""
    return settings
