import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

import src.config
from tests.test_utils import assert_equal


@pytest.mark.usefixtures("unmocked_config_settings")
def test_settings_initialization():
    from src.config import Settings
    settings = Settings(
        DATABASE_URL="postgresql://user:pass@localhost/db",
        REDIS_URL="redis://localhost:6379/0",
        RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
        JWT_SECRET="test-secret"
    )
    assert_equal(settings.PROJECT_NAME, "Black-Scholes Advanced Option Pricing Platform")
    assert settings.ENVIRONMENT in ["dev", "staging", "prod"]


@pytest.mark.usefixtures("unmocked_config_settings")
def test_settings_validation():
    from src.config import Settings
    # Test valid expiration
    settings = Settings(
        ACCESS_TOKEN_EXPIRE_MINUTES=60,
        DATABASE_URL="postgresql://user:pass@localhost/db",
        REDIS_URL="redis://localhost:6379/0",
        RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
        JWT_SECRET="test-secret"
    )
    assert_equal(settings.ACCESS_TOKEN_EXPIRE_MINUTES, 60)

    # Test invalid expiration (should raise ValueError due to @field_validator if it's there)
    # Based on my previous refactor, I saw validators in config.py
    with pytest.raises(ValueError):
        Settings(
            ACCESS_TOKEN_EXPIRE_MINUTES=-1,
            DATABASE_URL="postgresql://user:pass@localhost/db",
            REDIS_URL="redis://localhost:6379/0",
            RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
            JWT_SECRET="test-secret"
        )


@pytest.mark.usefixtures("unmocked_config_settings")
def test_password_min_length_validation():
    from src.config import Settings
    with pytest.raises(ValueError):
        Settings(
            PASSWORD_MIN_LENGTH=5,
            DATABASE_URL="postgresql://user:pass@localhost/db",
            REDIS_URL="redis://localhost:6379/0",
            RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
            JWT_SECRET="test-secret"
        )

# Reuse the fixture from conftest.py which handles reload
@pytest.mark.usefixtures("unmocked_config_settings")
def test_validators_coverage():
    from src.config import Settings
    
    # Test valid defaults
    s = Settings(
        DATABASE_URL="postgresql://user:pass@localhost/db",
        REDIS_URL="redis://localhost:6379/0",
        RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
        JWT_SECRET="test-secret"
    )
    assert s.MFA_ENCRYPTION_KEY is not None
    
    # Test validate_mfa_encryption_key default in dev
    # (Should be covered by default init)
    
    # Test validate_environment
    with pytest.raises(ValueError):
        Settings(
            ENVIRONMENT="invalid",
            DATABASE_URL="postgresql://user:pass@localhost/db",
            REDIS_URL="redis://localhost:6379/0",
            RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
            JWT_SECRET="test-secret"
        )
        
    # Test validate_log_level
    with pytest.raises(ValueError):
        Settings(
            LOG_LEVEL="TRACE",
            DATABASE_URL="postgresql://user:pass@localhost/db",
            REDIS_URL="redis://localhost:6379/0",
            RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
            JWT_SECRET="test-secret"
        )
        
    # Test validate_token_expiration
    with pytest.raises(ValueError):
        Settings(
            ACCESS_TOKEN_EXPIRE_MINUTES=-1,
            DATABASE_URL="postgresql://user:pass@localhost/db",
            REDIS_URL="redis://localhost:6379/0",
            RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
            JWT_SECRET="test-secret"
        )
    
    # Test validate_refresh_expiration
    with pytest.raises(ValueError):
        Settings(
            REFRESH_TOKEN_EXPIRE_DAYS=-1,
            DATABASE_URL="postgresql://user:pass@localhost/db",
            REDIS_URL="redis://localhost:6379/0",
            RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
            JWT_SECRET="test-secret"
        )
        
    # Test validate_bcrypt_rounds
    with pytest.raises(ValueError):
        Settings(
            BCRYPT_ROUNDS=5,
            DATABASE_URL="postgresql://user:pass@localhost/db",
            REDIS_URL="redis://localhost:6379/0",
            RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
            JWT_SECRET="test-secret"
        )
    with pytest.raises(ValueError):
        Settings(
            BCRYPT_ROUNDS=20,
            DATABASE_URL="postgresql://user:pass@localhost/db",
            REDIS_URL="redis://localhost:6379/0",
            RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
            JWT_SECRET="test-secret"
        )
        
    # Test validate_password_min_length
    with pytest.raises(ValueError):
        Settings(
            PASSWORD_MIN_LENGTH=5,
            DATABASE_URL="postgresql://user:pass@localhost/db",
            REDIS_URL="redis://localhost:6379/0",
            RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
            JWT_SECRET="test-secret"
        )
        
    # Test parse_cors_origins
    s = Settings(
        CORS_ORIGINS=[" http://a.com ", " "],
        DATABASE_URL="postgresql://user:pass@localhost/db",
        REDIS_URL="redis://localhost:6379/0",
        RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
        JWT_SECRET="test-secret"
    )
    assert s.CORS_ORIGINS == ["http://a.com"]
    
    # Test parse_cors_origins invalid input
    s = Settings(
        CORS_ORIGINS=123,
        DATABASE_URL="postgresql://user:pass@localhost/db",
        REDIS_URL="redis://localhost:6379/0",
        RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
        JWT_SECRET="test-secret"
    )
    assert s.CORS_ORIGINS == []

    # Test valid MFA key (hits line 156)
    s = Settings(
        MFA_ENCRYPTION_KEY="valid-key",
        DATABASE_URL="postgresql://user:pass@localhost/db",
        REDIS_URL="redis://localhost:6379/0",
        RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
        JWT_SECRET="test-secret"
    )
    assert s.MFA_ENCRYPTION_KEY == "valid-key"

    # Test environment normalization (hits 173, 175)
    s = Settings(
        ENVIRONMENT="Production",
        DATABASE_URL="postgresql://user:pass@localhost/db",
        REDIS_URL="redis://localhost:6379/0",
        RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
        JWT_SECRET="test-secret"
    )
    assert s.ENVIRONMENT == "prod"
    s = Settings(
        ENVIRONMENT="Test",
        DATABASE_URL="postgresql://user:pass@localhost/db",
        REDIS_URL="redis://localhost:6379/0",
        RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
        JWT_SECRET="test-secret"
    )
    assert s.ENVIRONMENT == "dev"

    # Test MFA key missing in prod
    with patch.dict(os.environ, {"ENVIRONMENT": "prod"}):
        with pytest.raises(ValueError, match="MFA_ENCRYPTION_KEY must be set"):
            Settings(
                MFA_ENCRYPTION_KEY="",
                DATABASE_URL="postgresql://user:pass@localhost/db",
                REDIS_URL="redis://localhost:6379/0",
                RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
                JWT_SECRET="test-secret"
            )

    # Test JWT secret default in prod
    with patch.dict(os.environ, {"ENVIRONMENT": "prod"}):
        with pytest.raises(ValueError, match="JWT_SECRET must be changed"):
            Settings(
                JWT_SECRET="change-me-in-production",
                DATABASE_URL="postgresql://user:pass@localhost/db",
                REDIS_URL="redis://localhost:6379/0",
                RABBITMQ_URL="amqp://guest:guest@localhost:5672//"
            )


def test_fallback_initialization_no_reload(monkeypatch):
    """Test fallback by patching functions in place without reload."""
    # We need to patch get_settings in src.config
    
    def mock_get_settings_fail():
        # Raise ValidationError to trigger the first except block
        raise ValidationError.from_exception_data("mock", []) 
        
    monkeypatch.setattr(src.config, "get_settings", mock_get_settings_fail)
    
    # Also patch Settings class to fail when called in fallback
    # The fallback tries: settings = Settings(...)
    # We want this to raise Exception so it goes to "Last resort"
    
    def MockSettingsFail(*args, **kwargs):
        raise ValueError("Simulated failure in fallback Settings init")
        
    monkeypatch.setattr(src.config, "Settings", MockSettingsFail)
    
    # Run initialization
    src.config._initialize_settings()
    
    # Verify fallback to MagicMock
    assert isinstance(src.config.settings, MagicMock)
    assert src.config.settings.ENVIRONMENT == "dev"


def test_configure_logging_execution(monkeypatch):
    # Mock sys.modules to exclude pytest to trigger configure_logging call
    with patch.dict(sys.modules):
        if "pytest" in sys.modules:
            del sys.modules["pytest"]
            
        with patch("src.config.configure_logging") as mock_log:
            # We also need get_settings to succeed or return something
            # It's currently mocked by pytest_configure to return mock_settings
            # That's fine.
            src.config._initialize_settings()
            
            mock_log.assert_called_once()
