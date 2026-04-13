import base64
import hashlib

import pytest

from src.shared.config import _DEFAULT_DEV_MFA_KEY, Settings


def test_secret_derivation_from_better_auth_secret():
    """Test that secrets are derived deterministically from BETTER_AUTH_SECRET."""
    master_secret = "test-master-secret-at-least-32-chars-long-123"
    settings = Settings(
        BETTER_AUTH_SECRET=master_secret,
        DATABASE_URL="postgresql://user:pass@localhost/db",
        REDIS_URL="redis://localhost:6379/0",
        RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
        JWT_SECRET="test-secret-at-least-32-chars-long-123",
    )

    # Verify MFA key derivation
    expected_mfa_seed = hashlib.sha256(f"mfa-derivation-{master_secret}".encode()).digest()
    expected_mfa_key = base64.urlsafe_b64encode(expected_mfa_seed).decode()
    assert settings.MFA_ENCRYPTION_KEY == expected_mfa_key
    assert settings.MFA_ENCRYPTION_KEY != _DEFAULT_DEV_MFA_KEY

    # Verify JWT secret derivation
    expected_jwt_seed = hashlib.sha256(f"jwt-derivation-{master_secret}".encode()).hexdigest()
    assert settings.JWT_SECRET == expected_jwt_seed


def test_explicit_secrets_override_derivation():
    """Test that explicitly provided secrets are not overridden by derivation."""
    master_secret = "test-master-secret-at-least-32-chars-long-123"
    explicit_mfa = base64.urlsafe_b64encode(b"explicit-mfa-key-32-bytes-long-12").decode()
    explicit_jwt = "explicit-jwt-secret"

    settings = Settings(
        BETTER_AUTH_SECRET=master_secret,
        MFA_ENCRYPTION_KEY=explicit_mfa,
        JWT_SECRET=explicit_jwt,
        DATABASE_URL="postgresql://user:pass@localhost/db",
        REDIS_URL="redis://localhost:6379/0",
        RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
        ENVIRONMENT="dev",
    )

    assert settings.MFA_ENCRYPTION_KEY == explicit_mfa
    assert settings.JWT_SECRET == explicit_jwt


def test_production_requires_robust_better_auth_secret():
    """Test that production environment requires robust BETTER_AUTH_SECRET."""

    # Missing secret
    with pytest.raises(ValueError, match="BETTER_AUTH_SECRET must be set in production"):
        Settings(
            ENVIRONMENT="prod",
            BETTER_AUTH_SECRET="",
            DATABASE_URL="postgresql://user:pass@localhost/db",
            REDIS_URL="redis://localhost:6379/0",
            RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
        )

    # Too short secret
    with pytest.raises(ValueError, match="BETTER_AUTH_SECRET must be at least 32 characters"):
        Settings(
            ENVIRONMENT="prod",
            BETTER_AUTH_SECRET="too-short",
            DATABASE_URL="postgresql://user:pass@localhost/db",
            REDIS_URL="redis://localhost:6379/0",
            RABBITMQ_URL="amqp://guest:guest@localhost:5672//",
        )