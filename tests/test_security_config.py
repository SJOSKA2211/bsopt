import pytest
from pydantic import ValidationError

from src.config import DEFAULT_DEV_MFA_KEY, Settings


def test_mfa_key_security_dev():
    """Test that default key is allowed in dev environment."""
    settings = Settings(
        ENVIRONMENT="dev",
        MFA_ENCRYPTION_KEY=DEFAULT_DEV_MFA_KEY,
        DATABASE_URL="postgresql://user:pass@localhost/db",
        REDIS_URL="redis://localhost:6379/0",
        JWT_SECRET="test-secret",
    )
    assert settings.MFA_ENCRYPTION_KEY == DEFAULT_DEV_MFA_KEY
    assert settings.ENVIRONMENT == "dev"


def test_mfa_key_security_prod_failure():
    """Test that default key is REJECTED in prod environment."""
    with pytest.raises(ValidationError) as excinfo:
        Settings(
            ENVIRONMENT="prod",
            MFA_ENCRYPTION_KEY=DEFAULT_DEV_MFA_KEY,
            DATABASE_URL="postgresql://user:pass@localhost/db",
            REDIS_URL="redis://localhost:6379/0",
            JWT_SECRET="test-secret",
        )
    assert "CRITICAL SECURITY ERROR" in str(excinfo.value)


def test_mfa_key_security_prod_success():
    """Test that CUSTOM key is allowed in prod environment."""
    custom_key = "custom_secure_key_for_production_use_only="
    settings = Settings(
        ENVIRONMENT="prod",
        MFA_ENCRYPTION_KEY=custom_key,
        DATABASE_URL="postgresql://user:pass@localhost/db",
        REDIS_URL="redis://localhost:6379/0",
        JWT_SECRET="test-secret",
    )
    assert settings.MFA_ENCRYPTION_KEY == custom_key
    assert settings.ENVIRONMENT == "prod"
