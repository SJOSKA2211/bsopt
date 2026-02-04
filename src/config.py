"""
Application configuration management.

Centralized configuration using Pydantic Settings with environment variable support.
Validates all configuration parameters at startup to fail fast on misconfiguration.
"""

import logging
import sys
import os
from enum import Enum
from typing import Any, List, Optional, Union, cast, Dict

import structlog
from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from src.shared.observability import setup_logging

logger = structlog.get_logger(__name__)

class SecretsProvider:
    """Base class for secret providers."""
    def get_secret(self, key: str) -> Optional[str]:
        return os.environ.get(key)

class SecretsManager:
    """
    Orchestrates secret retrieval from multiple providers.
    In production, this could be extended to use HashiCorp Vault, AWS Secrets Manager, etc.
    """
    def __init__(self):
        self.providers = [SecretsProvider()] # Fallback to ENV

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        for provider in self.providers:
            val = provider.get_secret(key)
            if val:
                return val
        return default

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and secret managers.
    """
    _secrets: SecretsManager = SecretsManager()

    # Application Configuration
    PROJECT_NAME: str = Field(
        default="Black-Scholes Advanced Option Pricing Platform", description="Application name"
    )
    ENVIRONMENT: str = Field(default="dev", description="Environment (dev, staging, prod)")
    DEBUG: bool = Field(default=True, description="Enable debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # Database Configuration
    DATABASE_URL: str = Field(
        validation_alias="DATABASE_URL",
        description="PostgreSQL database connection string",
    )

    # Redis Configuration
    REDIS_URL: str = Field(
        validation_alias="REDIS_URL",
        description="Redis connection string for caching and session management",
    )
    REDIS_HOST: str = Field(default="localhost", description="Redis host")
    REDIS_PORT: int = Field(default=6379, description="Redis port")
    REDIS_DB: int = Field(default=0, description="Redis database index")
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Redis password")

    # RabbitMQ Configuration
    RABBITMQ_URL: str = Field(
        validation_alias="RABBITMQ_URL",
        description="RabbitMQ connection string for async task processing",
    )

    # Rate Limiting Tiers
    RATE_LIMIT_FREE: int = 100
    RATE_LIMIT_PRO: int = 1000
    RATE_LIMIT_ENTERPRISE: int = 10000

    # JWT Authentication
    JWT_SECRET: str = Field(validation_alias="JWT_SECRET", description="Secret key for secondary JWT/HMAC operations")
    MFA_ENCRYPTION_KEY: Optional[str] = Field(
        description="Key for encrypting MFA secrets. Required in production/staging."
    )
    JWT_ALGORITHM: str = Field(default="RS256", description="JWT signing algorithm")

    JWT_PRIVATE_KEY: Optional[str] = Field("", description="Private key for JWT signing")
    JWT_PUBLIC_KEY: Optional[str] = Field("", description="Public key for JWT verification")
    
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

    # Argon2id Hashing Parameters (OWASP Recommended)
    ARGON2_TIME_COST: int = Field(default=3, description="Argon2 time cost (iterations)")
    ARGON2_MEMORY_COST: int = Field(default=65536, description="Argon2 memory cost (KiB)")
    ARGON2_PARALLELISM: int = Field(default=4, description="Argon2 parallelism (threads)")

    # ML Serving Configuration
    ML_SERVICE_URL: str = Field(
        default="http://localhost:8080",
        description="URL of the machine learning serving microservice",
    )
    ML_SERVICE_GRPC_URL: str = Field(
        default="localhost:50051",
        description="gRPC URL of the machine learning serving microservice",
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
    SENDGRID_API_KEY: Optional[str] = Field(
        default=None, description="SendGrid API Key for transactional emails"
    )
    DEFAULT_FROM_EMAIL: str = Field(
        default="noreply@example.com", description="Default sender email address for notifications"
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
    ML_TRAINING_NN_HIDDEN_DIMS: List[int] = Field(default=[128, 64, 32], description="Hidden layer dimensions for Neural Network")
    ML_TRAINING_DEFAULT_SAMPLES: int = Field(default=10000, description="Default number of samples for synthetic data generation")
    
    # MLflow tracking URI (pointing to local file by default, override in prod)
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    
    @property
    def tracking_uri(self) -> str:
        """🚀 SINGULARITY: Point MLflow to Postgres in production."""
        if self.ENVIRONMENT == "prod":
            # Strip asyncpg prefix if present as MLflow needs standard psycopg2
            url = self.DATABASE_URL.replace("postgresql+asyncpg", "postgresql")
            return url
        return self.MLFLOW_TRACKING_URI
    ML_RESULTS_DIR: str = Field(default="results/ml", description="Directory for ML results and artifacts")

    # XGBoost Default Configuration
    ML_XGBOOST_MAX_DEPTH: int = Field(default=6, description="XGBoost max depth")
    ML_XGBOOST_LEARNING_RATE: float = Field(default=0.1, description="XGBoost learning rate")
    ML_XGBOOST_N_ESTIMATORS: int = Field(default=100, description="XGBoost number of estimators")

    HESTON_MODEL_ONNX_PATH: Optional[str] = Field(default="models/heston_calibration.onnx", description="Path to the ONNX model for Heston calibration")
    OPTUNA_STORAGE_URL: Optional[str] = Field(
        default=None, 
        validation_alias="OPTUNA_STORAGE_URL",
        description="Database URL for Optuna storage to enable distributed HPO. Defaults to DATABASE_URL if None."
    )

    # Dask Configuration for Distributed ML Training
    DASK_LOCAL_CLUSTER_THREADS_PER_WORKER: int = Field(
        default_factory=lambda: max(1, os.cpu_count() // 2 if os.cpu_count() else 2),
        description="Number of threads per worker in local Dask cluster. Defaults to half of CPU count."
    )
    DASK_ARRAY_DEFAULT_CHUNKS_FRACTION: int = Field(default=4, description="Fraction of data length to use for Dask array chunk size (e.g., 4 means len(X)//4)")

    # Pricing Engine Configuration
    MONTE_CARLO_GPU_THRESHOLD: int = Field(default=500000, description="Minimum number of paths to trigger GPU acceleration")
    PRICING_LARGE_BATCH_THRESHOLD: int = Field(default=5000, description="Threshold for batch size to trigger parallel/Ray execution")
    PRICING_HESTON_PARALLEL_THRESHOLD: int = Field(default=2, description="Threshold for Heston batch size to trigger parallel execution")

    # Scraper Configuration
    NSE_CACHE_TTL: int = Field(default=60, description="Cache TTL for NSE data in seconds")
    NSE_SECTORS: List[str] = Field(
        default=["agric", "auto", "bank", "comm", "const", "energy", "insr", "invest", "investse", "manu", "tele", "real", "exchange"],
        description="List of NSE sectors to scrape"
    )
    NSE_NAME_SYMBOL_MAP: Dict[str, str] = Field(
        default={
            "SAFARICOM": "SCOM",
            "EQUITY": "EQTY",
            "KCB": "KCB",
            "ABSA": "ABSA",
            "CO-OP": "COOP",
            "BAT": "BAT",
            "EABL": "EABL",
            "KENGEN": "KEGN",
            "CENTUM": "CTUM",
            "BAMBURI": "BAMB",
            "NCBA": "NCBA",
            "SCBK": "SCBK"
        },
        description="Mapping of company names to symbols for NSE"
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    @model_validator(mode="after")
    def _fetch_external_secrets(self) -> "Settings":
        """
        Post-init hook to fetch sensitive values from SecretsManager.
        This allows secrets to be injected from external providers (Vault, etc.)
        even if they aren't in the .env file.
        """
        sensitive_fields = [
            "DATABASE_URL", "REDIS_PASSWORD", "JWT_SECRET", 
            "MFA_ENCRYPTION_KEY", "JWT_PRIVATE_KEY", "SENDGRID_API_KEY"
        ]
        
        for field in sensitive_fields:
            # Only override if the current value is empty or default
            current_val = getattr(self, field, None)
            if not current_val or current_val == "":
                secret = self._secrets.get(field)
                if secret:
                    setattr(self, field, secret)
                    
        return self

    @field_validator("MFA_ENCRYPTION_KEY", mode="before")
    @classmethod
    def validate_mfa_encryption_key(cls, v: Any, info) -> str:
        """Validate MFA encryption key or provide a default for tests."""
        env = os.environ.get("ENVIRONMENT", "dev").lower()
        if not v or v == "":
            if env in ["prod", "production", "staging"]:
                raise ValueError("MFA_ENCRYPTION_KEY must be set in production/staging environments")
            return ""
        return str(v)

    @field_validator("JWT_PRIVATE_KEY", "JWT_PUBLIC_KEY", mode="before")
    @classmethod
    def validate_jwt_keys(cls, v: Any, info) -> str:
        """Strictly enforce presence of RSA keys in production."""
        env = os.environ.get("ENVIRONMENT", "dev").lower()
        if not v or v == "":
            if env in ["prod", "production", "staging"]:
                # Accessing field name via info.field_name
                raise ValueError(f"{info.field_name} must be provided in production")
            return ""
        return str(v)


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

def _initialize_settings():
    global _settings, settings
    
    # -------------------------------------------------------------------------
    # AUTOMATIC THREAD TUNING FOR HIGH-PERFORMANCE COMPUTING
    # -------------------------------------------------------------------------
    # Optimization: Prevent thread oversubscription in Numba/NumPy.
    # If not explicitly set, default to physical core count.
    # This avoids context switching overhead when both parallel Numba and
    # OpenBLAS/MKL try to spawn threads.
    if not os.environ.get("OMP_NUM_THREADS"):
        # Use physical cores if available, otherwise logical
        try:
            num_threads = len(os.sched_getaffinity(0))
        except AttributeError:
            num_threads = os.cpu_count() or 1
            
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["NUMBA_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)

    # -------------------------------------------------------------------------
    # HARDWARE AWARE OPTIMIZATION (AVX-512 / GPU)
    # -------------------------------------------------------------------------
    try:
        # Check for AVX-512
        if sys.platform == "linux" and os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                if "avx512" in cpuinfo.lower():
                    os.environ["NUMBA_ENABLE_AVX"] = "1"
                    os.environ["NUMBA_ENABLE_AVX512"] = "1"
                    os.environ["NUMBA_OPT"] = "3"  # Aggressive vectorization
                    # print("Hardware Optimization: AVX-512 Detected & Enabled")

        # Check for NVIDIA GPU
        # Simple check: existence of nvidia-smi or /dev/nvidia0
        if os.path.exists("/dev/nvidia0"):
            os.environ["ONNX_PROVIDERS"] = "CUDAExecutionProvider,CPUExecutionProvider"
            # print("Hardware Optimization: NVIDIA GPU Detected. ONNX set to CUDA.")
        else:
            os.environ["ONNX_PROVIDERS"] = "CPUExecutionProvider"
            
    except Exception as e:
        # Don't crash on hardware detection failure
        pass
    # -------------------------------------------------------------------------

    try:
        _settings = get_settings()
        settings = _settings
        # Only configure logging if not in test mode
        if "pytest" not in sys.modules and not os.environ.get("BSOPT_TEST_MODE"):
            configure_logging(_settings)
    except Exception as e:
        # If configuration fails in a non-test environment, we must exit to avoid 
        # running in an undefined/insecure state.
        if "pytest" not in sys.modules and os.environ.get("ENVIRONMENT") in ["prod", "production", "staging"]:
            print(f"CRITICAL: Settings initialization failed in {os.environ.get('ENVIRONMENT')} environment: {e}")
            sys.exit(1)
            
        if "pytest" in sys.modules or os.environ.get("BSOPT_TEST_MODE"):
            logger.warning(f"Settings initialization failed, using safe defaults for tests: {e}")
            try:
                settings = Settings(
                    DATABASE_URL="sqlite:///:memory:",
                    REDIS_URL="redis://localhost:6379/0",
                    RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
                    JWT_SECRET="safe-default-secret-not-for-production",
                    MFA_ENCRYPTION_KEY=None,
                    JWT_PRIVATE_KEY=None,
                    JWT_PUBLIC_KEY=None
                )
                _settings = settings
                return
            except Exception as e2:
                logger.error(f"Failed to even initialize default settings: {e2}")
        
        logger.warning(f"Settings initialization failed. Please check environment variables. Error: {e}")
        raise

_initialize_settings()
