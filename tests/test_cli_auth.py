from unittest.mock import MagicMock, mock_open, patch

from src.cli.auth import AuthManager


def test_cli_login():
    with patch("httpx.Client") as mock_client_cls:
        # Setup the mock instance
        mock_instance = mock_client_cls.return_value
        # Ensure context manager returns the same instance
        mock_instance.__enter__.return_value = mock_instance

        # Setup response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "token",
            "refresh_token": "refresh",
            "expires_in": 3600,
            "user_id": "123",
            "email": "user@example.com",
            "tier": "free",
        }
        mock_instance.post.return_value = mock_response

        with patch("pathlib.Path.mkdir"):  # Avoid creating directories
            auth = AuthManager()

            # Mock file writing using mock_open
            with patch("builtins.open", mock_open()):
                result = auth.login("user@example.com", "password")

            assert result["access_token"] == "token"


def test_cli_logout():
    with patch("pathlib.Path.mkdir"):  # Avoid creating directories
        auth = AuthManager()
        # Mock the token_file Path object
        auth.token_file = MagicMock()
        auth.token_file.exists.return_value = True

        success = auth.logout()

        assert success
        auth.token_file.unlink.assert_called_once()
