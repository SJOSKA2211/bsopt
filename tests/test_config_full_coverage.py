import pytest
import sys
import os
import importlib
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

# We need to use unmocked settings to test the actual validators
@pytest.fixture
def unmocked_settings_module(monkeypatch):
    # Save mocks
    import src.config
    mocked_settings_instance = src.config.settings
    mocked_Settings_class = src.config.Settings
    mocked_get_settings_func = src.config.get_settings
    
    # Remove from sys.modules
    if "src.config" in sys.modules:
        del sys.modules["src.config"]
        
    import src.config
    importlib.reload(src.config)
    
    yield src.config
    
    # Restore mocks
    src.config.settings = mocked_settings_instance
    src.config._settings = mocked_settings_instance
    src.config.Settings = mocked_Settings_class
    src.config.get_settings = mocked_get_settings_func

def test_validators_coverage(unmocked_settings_module):
    Settings = unmocked_settings_module.Settings
    
    # Test valid defaults
    s = Settings()
    assert s.MFA_ENCRYPTION_KEY is not None
    
    # Test validate_mfa_encryption_key default in dev
    # (Should be covered by default init)
    
    # Test validate_environment
    with pytest.raises(ValueError):
        Settings(ENVIRONMENT="invalid")
        
    # Test validate_log_level
    with pytest.raises(ValueError):
        Settings(LOG_LEVEL="TRACE")
        
    # Test validate_token_expiration
    with pytest.raises(ValueError):
        Settings(ACCESS_TOKEN_EXPIRE_MINUTES=-1)
    
    # Test validate_refresh_expiration
    with pytest.raises(ValueError):
        Settings(REFRESH_TOKEN_EXPIRE_DAYS=-1)
        
    # Test validate_bcrypt_rounds
    with pytest.raises(ValueError):
        Settings(BCRYPT_ROUNDS=5)
    with pytest.raises(ValueError):
        Settings(BCRYPT_ROUNDS=20)
        
    # Test validate_password_min_length
    with pytest.raises(ValueError):
        Settings(PASSWORD_MIN_LENGTH=5)
        
    # Test parse_cors_origins
    s = Settings(CORS_ORIGINS=[" http://a.com ", " "])
    assert s.CORS_ORIGINS == ["http://a.com"]
    
    # Test parse_cors_origins invalid input
    # Pydantic might handle type error before validator if type doesn't match Union[str, List[str]]
    # But if we pass something that is allowed by type but validator handles it?
    # The validator handles Any.
    s = Settings(CORS_ORIGINS=123)
    assert s.CORS_ORIGINS == []

def test_fallback_initialization(monkeypatch):
    """Test the fallback logic in _initialize_settings when get_settings fails."""
    
    # We need to simulate get_settings failing.
    # We can do this by mocking get_settings in src.config before _initialize_settings is called.
    
    # 1. Unload src.config
    if "src.config" in sys.modules:
        del sys.modules["src.config"]
        
    # 2. Prepare import such that get_settings raises error
    # We can't easily mock a function inside the module we are about to import 
    # unless we use a custom loader or patch it immediately after import but before _initialize_settings.
    # But _initialize_settings calls get_settings immediately.
    
    # Alternative: Mock Settings class construction to fail inside get_settings?
    # get_settings calls Settings().
    # If we patch Settings to raise ValidationError.
    
    # Let's try to patch Pydantic BaseSettings or something used by Settings
    # or just let it run naturally and assume it passes, then we can't test fallback.
    
    # We can force a validation error by setting an invalid environment variable
    monkeypatch.setenv("ENVIRONMENT", "INVALID_ENV_FOR_FALLBACK")
    
    import src.config
    importlib.reload(src.config)
    
    # Now src.config.settings should be the fallback mock object
    # Verify it is a MagicMock (or whatever fallback creates)
    # The fallback creates:
    # settings = Settings( ... ) with dummy values.
    # If Settings() also fails (e.g. because of invalid env var), then it goes to "Last resort".
    # Since we set INVALID_ENV, Settings() will fail validation.
    # So it should hit the `except ValidationError` block in `get_settings`.
    # Wait, `get_settings` catches ValidationError and raises it.
    # `_initialize_settings` catches `Exception`.
    
    # So:
    # 1. `_initialize_settings` calls `get_settings()`
    # 2. `get_settings()` calls `Settings()`
    # 3. `Settings()` reads `ENVIRONMENT="INVALID_ENV_FOR_FALLBACK"` -> raises `ValidationError`
    # 4. `get_settings()` catches `ValidationError`, logs it, and re-raises it.
    # 5. `_initialize_settings` catches `Exception` (ValidationError is Exception).
    # 6. `_initialize_settings` enters `except Exception:` block.
    # 7. It tries `settings = Settings(MFA_ENCRYPTION_KEY=..., ...)` (Manual construction).
    #    This manual construction MIGHT still read env vars and fail?
    #    Pydantic settings read env vars.
    #    Yes, `Settings()` will still read `ENVIRONMENT` env var and fail!
    #    So the first fallback attempts to create `Settings(...)`. This will also fail due to `INVALID_ENV_FOR_FALLBACK`.
    # 8. The `try...except` block inside the fallback catches this failure.
    # 9. It goes to `# Last resort: use a mock`.
    # 10. `settings = MagicMock()`
    
    # So we expect src.config.settings to be a MagicMock
    
    assert isinstance(src.config.settings, MagicMock)
    assert src.config.settings.ENVIRONMENT == "test"
    
    # Clean up
    monkeypatch.delenv("ENVIRONMENT", raising=False)
