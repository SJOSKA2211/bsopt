import os
import pytest
import logging
import importlib
from pydantic import ValidationError
import src.config
from src.config import Settings, get_settings, configure_logging

def test_settings_validators():
    # Test valid values
    s = Settings(ENVIRONMENT="prod", LOG_LEVEL="DEBUG")
    assert s.ENVIRONMENT == "prod"
    assert s.LOG_LEVEL == "DEBUG"
    
    # Test invalid values to hit raise lines
    with pytest.raises(ValidationError):
        Settings(ENVIRONMENT="invalid")
        
    with pytest.raises(ValidationError):
        Settings(LOG_LEVEL="TRACE")
        
    with pytest.raises(ValidationError):
        Settings(ACCESS_TOKEN_EXPIRE_MINUTES=0)
        
    with pytest.raises(ValidationError):
        Settings(REFRESH_TOKEN_EXPIRE_DAYS=-1)
        
    with pytest.raises(ValidationError):
        Settings(BCRYPT_ROUNDS=9)
        
    with pytest.raises(ValidationError):
        Settings(PASSWORD_MIN_LENGTH=7)

def test_parse_cors_origins():
    # String input
    s = Settings(CORS_ORIGINS="http://a.com, http://b.com")
    assert s.CORS_ORIGINS == ["http://a.com", "http://b.com"]
    
    # List input
    s2 = Settings(CORS_ORIGINS=[" http://c.com ", ""])
    assert s2.CORS_ORIGINS == ["http://c.com"]
    
    # Invalid input (should return empty list)
    s3 = Settings(CORS_ORIGINS=123)
    assert s3.CORS_ORIGINS == []

def test_settings_properties():
    s_dev = Settings(ENVIRONMENT="dev")
    assert s_dev.is_development is True
    assert s_dev.is_production is False
    
    s_prod = Settings(ENVIRONMENT="prod")
    assert s_prod.is_production is True
    assert s_prod.is_development is False
    
    tiers = s_dev.rate_limit_tiers
    assert tiers["free"] == s_dev.RATE_LIMIT_FREE

def test_configure_logging():
    s = Settings(LOG_LEVEL="DEBUG", DEBUG=False)
    configure_logging(s)
    # Verify uvicorn/fastapi log levels were set
    assert logging.getLogger("uvicorn").level == logging.WARNING
    
    s_debug = Settings(LOG_LEVEL="DEBUG", DEBUG=True)
    configure_logging(s_debug)
    # Should skip setting uvicorn/fastapi levels (they stay as they were or we don't care, just hit the False branch)

def test_get_settings_cache(tmp_path, monkeypatch):
    importlib.reload(src.config)
    real_get_settings = src.config.get_settings
    
    priv_key = tmp_path / "priv.pem"
    pub_key = tmp_path / "pub.pem"
    priv_key.write_text("private")
    pub_key.write_text("public")
    
    monkeypatch.setattr(src.config, "_settings", None)
    monkeypatch.setenv("JWT_PRIVATE_KEY_PATH", str(priv_key))
    monkeypatch.setenv("JWT_PUBLIC_KEY_PATH", str(pub_key))
    
    s1 = real_get_settings()
    s2 = real_get_settings() # Should hit the cache (line 239)
    assert s1 is s2

def test_get_settings_real_implementation(tmp_path, monkeypatch):
    # Restore the real get_settings if it was mocked in conftest.py
    # We need to reach the actual code in src/config.py
    
    # Save original functions/state
    orig_get_settings = get_settings
    
    # Reload src.config to get real implementation
    # But wait, conftest.py might have replaced it in a way that reload doesn't fix if it's still patching
    # Actually, let's just call the module-level function directly if we can get it
    importlib.reload(src.config)
    real_get_settings = src.config.get_settings
    
    # Create dummy key files
    priv_key = tmp_path / "priv.pem"
    pub_key = tmp_path / "pub.pem"
    priv_key.write_text("private")
    pub_key.write_text("public")
    
    monkeypatch.setattr(src.config, "_settings", None)
    monkeypatch.setenv("JWT_PRIVATE_KEY_PATH", str(priv_key))
    monkeypatch.setenv("JWT_PUBLIC_KEY_PATH", str(pub_key))
    # Ensure it doesn't fail on other required env vars if any
    
    try:
        s = real_get_settings()
        assert s.JWT_PRIVATE_KEY == "private"
        assert s.JWT_PUBLIC_KEY == "public"
    finally:
        # Restore mock for other tests
        src.config.get_settings = orig_get_settings

def test_get_settings_file_not_found(tmp_path, monkeypatch):
    importlib.reload(src.config)
    real_get_settings = src.config.get_settings
    
    monkeypatch.setattr(src.config, "_settings", None)
    monkeypatch.setenv("JWT_PRIVATE_KEY_PATH", str(tmp_path / "nonexistent"))
    
    with pytest.raises(FileNotFoundError):
        real_get_settings()

def test_get_settings_validation_error(monkeypatch):
    importlib.reload(src.config)
    real_get_settings = src.config.get_settings
    
    monkeypatch.setattr(src.config, "_settings", None)
    monkeypatch.setenv("BCRYPT_ROUNDS", "5")
    
    with pytest.raises(ValidationError):
        real_get_settings()
