import pytest
import os
from src.config import Settings, get_settings, configure_logging
from pydantic import ValidationError

def test_settings_validation():
    # Valid
    s = Settings(ENVIRONMENT="prod", LOG_LEVEL="DEBUG")
    assert s.ENVIRONMENT == "prod"
    assert s.LOG_LEVEL == "DEBUG"
    assert s.is_production
    assert not s.is_development

def test_settings_invalid_env():
    with pytest.raises(ValidationError):
        Settings(ENVIRONMENT="invalid")

def test_settings_invalid_log_level():
    with pytest.raises(ValidationError):
        Settings(LOG_LEVEL="TRACE")

def test_settings_invalid_token_expiry():
    with pytest.raises(ValidationError):
        Settings(ACCESS_TOKEN_EXPIRE_MINUTES=0)

def test_settings_invalid_refresh_expiry():
    with pytest.raises(ValidationError):
        Settings(REFRESH_TOKEN_EXPIRE_DAYS=-1)

def test_settings_invalid_bcrypt_rounds():
    with pytest.raises(ValidationError):
        Settings(BCRYPT_ROUNDS=9)
    with pytest.raises(ValidationError):
        Settings(BCRYPT_ROUNDS=15)

def test_settings_invalid_password_length():
    with pytest.raises(ValidationError):
        Settings(PASSWORD_MIN_LENGTH=7)

def test_parse_cors_origins():
    # String
    s = Settings(CORS_ORIGINS="a, b, c")
    assert s.CORS_ORIGINS == ["a", "b", "c"]
    
    # List
    s2 = Settings(CORS_ORIGINS=["x", "y "])
    assert s2.CORS_ORIGINS == ["x", "y"]
    
    # Invalid
    s3 = Settings(CORS_ORIGINS=123)
    assert s3.CORS_ORIGINS == []

def test_rate_limit_tiers():
    s = Settings()
    tiers = s.rate_limit_tiers
    assert tiers["free"] == 100

def test_get_settings_success(mocker):
    import importlib
    import src.config
    importlib.reload(src.config) # Reload to clear conftest patches for this test
    
    # Force reset of the singleton
    mocker.patch("src.config._settings", None)
    
    # Mock file reading for JWT keys
    m_open = mocker.patch("builtins.open", mocker.mock_open(read_data="mock_key"))
    
    # Mock environment variables to override paths AND keys
    # Use patch.dict on os.environ directly to ensure Pydantic sees it
    mocker.patch.dict(os.environ, {
        "ENVIRONMENT": "dev",
        "JWT_PRIVATE_KEY_PATH": "mock_priv",
        "JWT_PUBLIC_KEY_PATH": "mock_pub",
        "JWT_PRIVATE_KEY": "", 
        "JWT_PUBLIC_KEY": "",
        "BSOPT_TEST_MODE": "true"
    })
    
    settings = get_settings()
    assert settings.JWT_PRIVATE_KEY == "mock_key"
    
    # Check cache
    mocker.patch("src.config._settings", settings)
    assert get_settings() is settings

def test_get_settings_file_not_found(mocker):
    import importlib
    import src.config
    importlib.reload(src.config)

    # Reset singleton
    mocker.patch("src.config._settings", None)
    
    # Mock environment to avoid validation error and ensure it tries to read files
    mocker.patch.dict(os.environ, {
        "ENVIRONMENT": "dev",
        "JWT_PRIVATE_KEY": "",
        "JWT_PUBLIC_KEY": "",
        "BSOPT_TEST_MODE": "true"
    })
    
    # Mock open to raise FileNotFoundError
    mocker.patch("builtins.open", side_effect=FileNotFoundError("Mocked file not found"))
    
    with pytest.raises(FileNotFoundError):
        get_settings()

def test_get_settings_validation_error(mocker):
    import importlib
    import src.config
    importlib.reload(src.config)

    # Reset singleton
    mocker.patch("src.config._settings", None)
    
    # Force validation error by passing invalid env
    mocker.patch.dict(os.environ, {"ENVIRONMENT": "invalid", "BSOPT_TEST_MODE": "true"})
    
    with pytest.raises(ValidationError):
        get_settings()

def test_settings_environment_edge_cases():
    s1 = Settings(ENVIRONMENT="production")
    assert s1.ENVIRONMENT == "prod"
    s2 = Settings(ENVIRONMENT="test")
    assert s2.ENVIRONMENT == "dev"

def test_configure_logging():
    s = Settings(LOG_LEVEL="ERROR", DEBUG=False)
    configure_logging(s)
    # Just check if it runs without error
