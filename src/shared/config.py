import os
from typing import Annotated

import structlog
from pydantic import AliasChoices, BeforeValidator, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)

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
        default="postgresql://admin:password@postgres:5432/bsopt",
        validation_alias="DATABASE_URL",
    )
    DATABASE_MIN_POOL_SIZE: int = 5
    DATABASE_MAX_POOL_SIZE: int = 15
    DATABASE_POOL_TIMEOUT: int = 60
    DATABASE_POOL_RECYCLE: int = 1800
    DATABASE_POOL_PRE_PING: bool = True
    SLOW_QUERY_THRESHOLD_MS: int = 100
    PGBOUNCER_ENABLED: bool = Field(default=False, validation_alias="PGBOUNCER_ENABLED")

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v.startswith("postgresql://") and not v.startswith("postgresql+asyncpg://"):
            if "sqlite" not in v:
                raise ValueError("DATABASE_URL must be a valid PostgreSQL connection string.")
        return v

    # Redis Configuration
    REDIS_URL: str = Field(
        default="redis://:bsopt_redis_secret@redis:6379/0",
        validation_alias="REDIS_URL",
    )
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

    @property
    def CELERY_BROKER_URL(self) -> str:
        """Constructs the Celery Broker URL from RabbitMQ credentials."""
        return self.RABBITMQ_URL

    # ML Serving Configuration
    ML_SERVICE_GRPC_URL: str = "worker:50051"

    # Security Configuration
    OPA_URL: str = Field(default="http://opa:8181/v1/data/authz/allow", validation_alias="OPA_URL")
    AUDIT_VAULT_KEY: str = Field(
        default="manifold-vault-key-base-v1", validation_alias="AUDIT_VAULT_KEY"
    )
    
    @field_validator("AUDIT_VAULT_KEY", "RABBITMQ_PASSWORD", "REDIS_PASSWORD", "BLOCKCHAIN_PRIVATE_KEY")
    @classmethod
    def validate_secret_strength(cls, v: str | None, info: Any) -> str | None:
        if v is None:
            return v
        if len(v) < 32 and not os.environ.get("BSOPT_ALLOW_WEAK_SECRETS"):
            raise ValueError(f"{info.field_name} must be at least 32 characters for production security.")
        return v

    # Blockchain Configuration
    BLOCKCHAIN_RPC_URL: str = Field(
        default="http://geth:8545", validation_alias="BLOCKCHAIN_RPC_URL"
    )
    BLOCKCHAIN_PRIVATE_KEY: str = Field(
        default="0x0000000000000000000000000000000000000000000000000000000000000000",
        validation_alias="BLOCKCHAIN_PRIVATE_KEY",
    )

    # IBM Quantum Configuration
    IBM_QUANTUM_TOKEN: str | None = Field(default=None, validation_alias="IBM_QUANTUM_TOKEN")

    # Pricing Configuration
    MONTE_CARLO_GPU_THRESHOLD: int = 10000
    PRICING_LARGE_BATCH_THRESHOLD: int = 1000
    MAX_NET_DELTA: float = 10000.0
    MAX_NET_GAMMA: float = 5000.0
    MAX_NET_VEGA: float = 5000.0

    # ML Training Configuration
    ML_TRAINING_DEFAULT_SAMPLES: int = 1000
    ML_TRAINING_OPTUNA_TRIALS: int = 50
    ML_TRAINING_RANDOM_STATE: int = 42
    ML_TRAINING_PROMOTE_THRESHOLD_R2: float = Field(default=0.95, validation_alias="ML_TRAINING_PROMOTE_THRESHOLD_R2")

    # Email Configuration
    EMAIL_SERVICE_API_KEY: str | None = Field(default=None, validation_alias="EMAIL_SERVICE_API_KEY")
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
    USE_GPU: bool = Field(default=False)

    # Observability & Tracing
    ENABLE_TRACING: bool = False
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://tempo:4317"
    PUSHGATEWAY_URL: str | None = Field(default=None)
    GRAFANA_URL: str | None = Field(default=None)
    CHAOS_MODE: bool = Field(default=False, validation_alias="BSOPT_CHAOS_MODE")

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
    MINIO_ENDPOINT: str = Field(default="minio:9000", validation_alias="MINIO_ENDPOINT")
    MINIO_ROOT_USER: str = Field(default="minio_admin", validation_alias="MINIO_ROOT_USER")
    MINIO_ROOT_PASSWORD: str = Field(default="minio_secret_key", validation_alias="MINIO_ROOT_PASSWORD")
    MINIO_USE_SSL: bool = Field(default=False, validation_alias="MINIO_USE_SSL")

    @property
    def MINIO_ENDPOINT_URL(self) -> str:
        """Constructs the MinIO endpoint URL."""
        protocol = "https" if self.MINIO_USE_SSL else "http"
        return f"{protocol}://{self.MINIO_ENDPOINT}"

    # JWT Authentication
    JWT_SECRET: str = Field(default="", validation_alias="JWT_SECRET")
    JWT_ALGORITHM: str = "RS256"
    JWT_PRIVATE_KEY: str | None = Field(
        default=None, validation_alias=AliasChoices("JWT_PRIVATE_KEY", "JWT_RS256_PRIVATE")
    )
    JWT_PUBLIC_KEY: str | None = Field(
        default=None, validation_alias=AliasChoices("JWT_PUBLIC_KEY", "JWT_RS256_PUBLIC")
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
        default=_DEFAULT_MFA_KEY_SEED or "placeholder-mfa-key-seed-base-v1",
        validation_alias="MFA_ENCRYPTION_KEY",
    )
    
    # E2E & Testing
    ALLOW_E2E_EMAIL_BYPASS: bool = Field(default=False, validation_alias="ALLOW_E2E_EMAIL_BYPASS")

    # Better Auth Configuration
    BETTER_AUTH_SECRET: str = Field(default="", validation_alias="BETTER_AUTH_SECRET")
    BETTER_AUTH_URL: str = Field(
        default="http://localhost:3001", validation_alias="BETTER_AUTH_URL"
    )

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
        """Point MLflow to Postgres always."""
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
        """Validate security-critical settings and apply environment defaults."""

        # 0. Master Secret and Key Derivation
        if self.is_production:
            if not self.BETTER_AUTH_SECRET:
                raise ValueError("CRITICAL: BETTER_AUTH_SECRET must be set in production.")
            if len(self.BETTER_AUTH_SECRET) < 32:
                raise ValueError("CRITICAL: BETTER_AUTH_SECRET must be at least 32 characters for robust key derivation.")

        # Derivation logic (Shared between dev and prod if master secret exists)
        if self.BETTER_AUTH_SECRET:
            import base64
            import hashlib

            # Key derivation (PBKDF2-HMAC-SHA256)
            salt = b"manifold-derivation-v1"
            iterations = 100_000

            def derive_key(purpose: str, length: int = 32) -> bytes:
                return hashlib.pbkdf2_hmac(
                    "sha256",
                    self.BETTER_AUTH_SECRET.encode(),
                    salt + purpose.encode(),
                    iterations,
                    length,
                )

            # Derive MFA Encryption Key if not explicitly set
            if not self.MFA_ENCRYPTION_KEY or self.MFA_ENCRYPTION_KEY == _DEFAULT_MFA_KEY_SEED:
                mfa_seed = derive_key("mfa-encryption")
                self.MFA_ENCRYPTION_KEY = base64.urlsafe_b64encode(mfa_seed).decode()
                logger.debug("derived_mfa_key")

            # Derive JWT Secret if not explicitly set
            if not self.JWT_SECRET or self.JWT_SECRET == "change-me-in-production":
                jwt_seed = derive_key("jwt-signing", length=64)
                self.JWT_SECRET = jwt_seed.hex()
                logger.debug("derived_jwt_secret")

        # Enforce email verification in production by default if not set
        if self.is_production and self.ENVIRONMENT != "test":
            pass

        if self.is_production:
            # 1. JWT Secret Security
            if self.JWT_SECRET == "change-me-in-production" or not self.JWT_SECRET:
                raise ValueError(
                    "CRITICAL: JWT_SECRET must be changed from the default or derived from BETTER_AUTH_SECRET in production."
                )

            # 2. MFA Key Security
            key = self.MFA_ENCRYPTION_KEY
            if not key or key == _DEFAULT_MFA_KEY_SEED:
                raise ValueError(
                    "CRITICAL: MFA_ENCRYPTION_KEY must be set or derived in production."
                )

            try:
                import base64

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

            # 3. Email Verification Security
            if not self.REQUIRE_EMAIL_VERIFICATION:
                logger.warning(
                    "security_warning: email verification is DISABLED in production",
                    environment=self.ENVIRONMENT,
                )

        return self

settings = Settings()

def get_settings() -> Settings:
    """Returns the singleton settings instance."""
    return settings
