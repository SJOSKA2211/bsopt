import pytest

from src.config import Settings
from tests.test_utils import assert_equal


def test_settings_initialization():
    settings = Settings()
    assert_equal(settings.PROJECT_NAME, "Black-Scholes Advanced Option Pricing Platform")
    assert settings.ENVIRONMENT in ["dev", "staging", "prod"]


def test_settings_validation():
    # Test valid expiration
    settings = Settings(ACCESS_TOKEN_EXPIRE_MINUTES=60)
    assert_equal(settings.ACCESS_TOKEN_EXPIRE_MINUTES, 60)

    # Test invalid expiration (should raise ValueError due to @field_validator if it's there)
    # Based on my previous refactor, I saw validators in config.py
    with pytest.raises(ValueError):
        Settings(ACCESS_TOKEN_EXPIRE_MINUTES=-1)


def test_password_min_length_validation():
    with pytest.raises(ValueError):
        Settings(PASSWORD_MIN_LENGTH=5)
