import pytest
from tests.test_utils import assert_equal
import src.config

@pytest.mark.usefixtures("unmocked_config_settings")
def test_settings_initialization():
    from src.config import Settings
    settings = Settings()
    assert_equal(settings.PROJECT_NAME, "Black-Scholes Advanced Option Pricing Platform")
    assert settings.ENVIRONMENT in ["dev", "staging", "prod"]


@pytest.mark.usefixtures("unmocked_config_settings")
def test_settings_validation():
    from src.config import Settings
    # Test valid expiration
    settings = Settings(ACCESS_TOKEN_EXPIRE_MINUTES=60)
    assert_equal(settings.ACCESS_TOKEN_EXPIRE_MINUTES, 60)

    # Test invalid expiration (should raise ValueError due to @field_validator if it's there)
    # Based on my previous refactor, I saw validators in config.py
    with pytest.raises(ValueError):
        Settings(ACCESS_TOKEN_EXPIRE_MINUTES=-1)


@pytest.mark.usefixtures("unmocked_config_settings")
def test_password_min_length_validation():
    from src.config import Settings
    with pytest.raises(ValueError):
        Settings(PASSWORD_MIN_LENGTH=5)
