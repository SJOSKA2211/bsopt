"""
Tests for MFA key security validation in Settings.
"""

import os

import pytest
from pydantic import ValidationError


def _make_settings(env_overrides: dict):
    """Helper to construct Settings with specific env vars."""
    # Patch environment, then import fresh settings class
    env = {
        "DATABASE_URL": "postgresql://admin:test@localhost:5432/bsopt",
        "REDIS_URL": "redis://localhost:6379/0",
        "JWT_SECRET": "test_secret",
        **env_overrides,
    }
    with pytest.MonkeyPatch.context() as mp:
        for k, v in env.items():
            mp.setenv(k, v)
        from src.config import Settings

        return Settings()


class TestMfaKeyDevEnvironment:
    """Default MFA key should be accepted in dev/test environments."""

    def test_default_key_allowed_in_dev(self):
        settings = _make_settings({"ENVIRONMENT": "dev"})
        assert settings.MFA_ENCRYPTION_KEY is not None

    def test_default_key_allowed_in_test(self):
        settings = _make_settings({"ENVIRONMENT": "test"})
        assert settings.MFA_ENCRYPTION_KEY is not None


class TestMfaKeyProdEnvironment:
    """Using the default dev key in production must raise a ValidationError."""

    def test_default_key_blocked_in_prod(self):
        with pytest.raises(ValidationError, match="CRITICAL"):
            _make_settings({"ENVIRONMENT": "prod"})

    def test_default_key_blocked_in_production(self):
        """Also test the 'production' ENVIRONMENT alias."""
        with pytest.raises(ValidationError, match="CRITICAL"):
            _make_settings({"ENVIRONMENT": "production"})

    def test_custom_key_accepted_in_prod(self):
        # A proper 32-byte base64url key
        import base64

        secure_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
        settings = _make_settings(
            {"ENVIRONMENT": "prod", "MFA_ENCRYPTION_KEY": secure_key}
        )
        assert settings.MFA_ENCRYPTION_KEY == secure_key

    def test_short_key_rejected_in_prod(self):
        """A key that decodes to fewer than 32 bytes must be rejected."""
        import base64

        short_key = base64.urlsafe_b64encode(b"tooshort").decode()
        with pytest.raises(ValidationError, match="too short"):
            _make_settings(
                {"ENVIRONMENT": "prod", "MFA_ENCRYPTION_KEY": short_key}
            )


class TestEnvironmentNormalization:
    """ENVIRONMENT values should be normalized to lowercase."""

    def test_prod_uppercase(self):
        """'Prod' should normalize to 'prod'."""
        import base64

        secure_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
        settings = _make_settings(
            {"ENVIRONMENT": "Prod", "MFA_ENCRYPTION_KEY": secure_key}
        )
        assert settings.ENVIRONMENT == "prod"

    def test_invalid_environment_rejected(self):
        with pytest.raises(ValidationError, match="ENVIRONMENT must be one of"):
            _make_settings({"ENVIRONMENT": "banana"})
