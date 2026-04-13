import uuid
from datetime import UTC, datetime
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from api.index import app
from api.schemas.auth import TokenResponse
from src.auth.auth import auth_service, get_current_active_user, get_current_user
from src.auth.core.tokens import TokenData
from src.database import get_async_db
from src.database.models import User

# Use raise_server_exceptions=False to test exception handlers
client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def mock_user():
    user = User(
        id=uuid.uuid4(),
        email="auth_test@example.com",
        hashed_password="hashed_password",
        full_name="Auth Test",
        tier="free",
        is_active=True,
        is_verified=True,
        mfa_enabled=False,
        created_at=datetime.now(UTC),
    )
    return user


@pytest.fixture
def override_auth_dependencies(mock_user):
    """Bypass auth for testing routes that require it."""
    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    
    # Also patch the middleware's auth service hop
    with patch.object(auth_service, "validate_token", new_callable=AsyncMock) as mock_val:
        mock_val.return_value = TokenData(
            user_id=str(mock_user.id),
            email=mock_user.email,
            tier=mock_user.tier,
            token_type="access",
            jti="jti123",
            exp=datetime.now(),
            iat=datetime.now()
        )
        yield mock_val
        
    app.dependency_overrides = {}


def test_login_success(mock_user):
    mock_db = AsyncMock()
    # Mock the native login query result
    mock_result = MagicMock()
    mock_result.fetchone.return_value = (mock_user.id, mock_user.email, mock_user.tier, mock_user.is_active)
    mock_db.execute.return_value = mock_result
    
    app.dependency_overrides[get_async_db] = lambda: mock_db
    
    with patch("api.routes.auth.auth_service.create_token_pair") as mock_tokens:
        mock_tokens.return_value = TokenResponse(
            access_token="access",
            refresh_token="refresh",
            token_type="bearer",
            expires_in=3600,
            user_id=str(mock_user.id),
            email=mock_user.email
        )
        payload = {"email": "auth_test@example.com", "password": "password123"}
        # Login is a public path, so ZeroTrustMiddleware skips it
        response = client.post("/api/v1/auth/login", json=payload)
        assert response.status_code == 200
        assert response.json()["data"]["access_token"] == "access"
    app.dependency_overrides = {}


def test_login_invalid_credentials():
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.fetchone.return_value = None # No user found
    mock_db.execute.return_value = mock_result
    
    app.dependency_overrides[get_async_db] = lambda: mock_db
    
    payload = {"email": "wrong@example.com", "password": "wrong"}
    response = client.post("/api/v1/auth/login", json=payload)
    assert response.status_code == 401
    app.dependency_overrides = {}


def test_register_success(mock_user):
    mock_db = AsyncMock()
    # SELECT register_user_native
    mock_reg_result = MagicMock()
    mock_reg_result.scalar.return_value = mock_user.id
    # SELECT User
    mock_user_result = MagicMock()
    mock_user_result.scalar_one.return_value = mock_user
    
    mock_db.execute.side_effect = [mock_reg_result, mock_user_result]
    
    app.dependency_overrides[get_async_db] = lambda: mock_db
    
    with (
        patch("api.routes.auth.auth_service.create_token_pair") as mock_tokens,
        patch("api.routes.auth._send_verification_email", new_callable=AsyncMock)
    ):
        mock_tokens.return_value = TokenResponse(
            access_token="access",
            token_type="bearer",
        )
        payload = {
            "email": "new@example.com",
            "password": "StrongPassword123!",
            "password_confirm": "StrongPassword123!",
            "full_name": "New User",
            "accept_terms": True
        }
        response = client.post("/api/v1/auth/register", json=payload)
        assert response.status_code == 201
        assert response.json()["data"]["access_token"] == "access"
    app.dependency_overrides = {}


def test_logout(override_auth_dependencies):
    with patch("api.routes.auth.auth_service.revoke_token", new_callable=AsyncMock) as mock_revoke:
        # Provide token to satisfy ZeroTrustMiddleware AND route logic
        response = client.post(
            "/api/v1/auth/logout", 
            headers={"Authorization": "Bearer some_token"}
        )
        assert response.status_code == 200
        mock_revoke.assert_called_once_with("some_token")


def test_me(override_auth_dependencies, mock_user):
    # Provide token to satisfy ZeroTrustMiddleware
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer some_token"}
    )
    assert response.status_code == 200
    assert response.json()["data"]["email"] == mock_user.email


def test_refresh_token_success(mock_user):
    with (
        patch("api.routes.auth.auth_service.decode_token") as mock_decode,
        patch("api.routes.auth.auth_service.token_blacklist.contains", new_callable=AsyncMock, return_value=False),
        patch("api.routes.auth.auth_service.token_blacklist.add", new_callable=AsyncMock),
        patch("api.routes.auth.auth_service.create_token_pair") as mock_create,
    ):
        mock_decode.return_value = MagicMock(
            user_id=str(mock_user.id), 
            email=mock_user.email,
            tier=mock_user.tier,
            token_type="refresh",
            jti="jti123",
            exp=datetime.now()
        )
        mock_create.return_value = TokenResponse(
            access_token="new_access",
            refresh_token="new_refresh",
            token_type="bearer",
            expires_in=3600
        )
        
        response = client.post("/api/v1/auth/refresh", json={"refresh_token": "old_refresh"})
        assert response.status_code == 200
        assert response.json()["data"]["access_token"] == "new_access"


def test_password_change_success(override_auth_dependencies, mock_user):
    mock_db = AsyncMock()
    app.dependency_overrides[get_async_db] = lambda: mock_db
    
    with (
        patch("api.routes.auth.auth_service.verify_password", return_value=True),
        patch("api.routes.auth.auth_service.hash_password", return_value="new_hash"),
    ):
        payload = {
            "current_password": "old",
            "new_password": "NewPassword123!",
            "new_password_confirm": "NewPassword123!",
        }
        response = client.post(
            "/api/v1/auth/password/change", 
            json=payload,
            headers={"Authorization": "Bearer some_token"}
        )
        assert response.status_code == 200
        assert mock_user.hashed_password == "new_hash"
        mock_db.commit.assert_called_once()


def test_mfa_setup_success(override_auth_dependencies, mock_user):
    mock_db = AsyncMock()
    app.dependency_overrides[get_async_db] = lambda: mock_db
    
    with (
        patch("api.routes.auth.auth_service.generate_mfa_secret", return_value="secret"),
        patch("api.routes.auth.auth_service.encrypt_mfa_secret", return_value="encrypted"),
        patch("api.routes.auth.auth_service.get_totp_uri", return_value="otpauth://..."),
    ):
        response = client.post(
            "/api/v1/auth/mfa/setup",
            headers={"Authorization": "Bearer some_token"}
        )
        assert response.status_code == 200
        assert response.json()["data"]["secret"] == "secret"
        assert mock_user.mfa_secret == "encrypted"


def test_mfa_verify_success(override_auth_dependencies, mock_user):
    mock_user.mfa_secret = "encrypted"
    mock_db = AsyncMock()
    app.dependency_overrides[get_async_db] = lambda: mock_db
    
    with (
        patch("api.routes.auth.auth_service.decrypt_mfa_secret", return_value="secret"),
        patch("api.routes.auth.auth_service.verify_mfa_code", return_value=True),
    ):
        response = client.post(
            "/api/v1/auth/mfa/verify", 
            json={"code": "123456"},
            headers={"Authorization": "Bearer some_token"}
        )
        assert response.status_code == 200
        assert mock_user.mfa_enabled is True
        mock_db.commit.assert_called_once()


def test_mfa_disable_success(override_auth_dependencies, mock_user):
    mock_user.mfa_enabled = True
    mock_user.mfa_secret = "encrypted"
    mock_db = AsyncMock()
    app.dependency_overrides[get_async_db] = lambda: mock_db
    
    with (
        patch("api.routes.auth.auth_service.decrypt_mfa_secret", return_value="secret"),
        patch("api.routes.auth.auth_service.verify_mfa_code", return_value=True),
    ):
        response = client.post(
            "/api/v1/auth/mfa/disable", 
            json={"code": "123456"},
            headers={"Authorization": "Bearer some_token"}
        )
        assert response.status_code == 200
        assert mock_user.mfa_enabled is False
        assert mock_user.mfa_secret is None
        mock_db.commit.assert_called_once()