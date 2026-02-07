from unittest.mock import MagicMock, patch

import pytest

from src.cli.auth import AuthManager


@pytest.fixture
def auth_manager(tmp_path):
    # Mock home directory for token file
    with patch("src.cli.auth.Path.home", return_value=tmp_path):
        manager = AuthManager(api_base_url="http://api")
        yield manager

def test_login_success(auth_manager):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "fake-token",
        "user": {"email": "test@example.com"}
    }
    
    with patch("httpx.Client.post", return_value=mock_response):
        result = auth_manager.login("test@example.com", "pass")
        assert result["access_token"] == "fake-token"
        assert auth_manager.token_file.exists()
        assert auth_manager.get_token() == "fake-token"

def test_logout(auth_manager):
    # Create fake token file
    auth_manager.token_file.parent.mkdir(parents=True, exist_ok=True)
    auth_manager.token_file.write_text("{}")
    
    assert auth_manager.logout() is True
    assert not auth_manager.token_file.exists()

def test_get_current_user_no_token(auth_manager):
    assert auth_manager.get_current_user() is None
