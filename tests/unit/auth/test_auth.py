import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
from api.index import app
from src.database import get_async_db
from src.database.models import User
from datetime import datetime, UTC
from uuid import uuid4

TEST_EMAIL = "test_auth_unique_2025@example.com"
TEST_PASSWORD = "Short_Secure_Pass_123!"
TEST_NAME = "Test User"

@pytest.fixture
def mock_db():
    mock = AsyncMock()
    return mock

@pytest.fixture(autouse=True)
def override_db(mock_db):
    app.dependency_overrides[get_async_db] = lambda: mock_db
    yield
    app.dependency_overrides.clear()

@pytest.fixture
def auth_data():
    return {
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD,
        "password_confirm": TEST_PASSWORD,
        "full_name": TEST_NAME,
        "accept_terms": True,
    }

client = TestClient(app, raise_server_exceptions=False)

def test_health_check():
    """Test if the API is running."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_register_success(mock_db, auth_data):
    """Test user registration."""
    # Mock the native DB call
    mock_db.execute.side_effect = [
        MagicMock(scalar=lambda: uuid4()), # First call: SELECT register_user_native
        MagicMock(scalar_one=lambda: User(id=uuid4(), email=TEST_EMAIL, tier="free")) # Second call: SELECT User
    ]
    
    response = client.post("/api/v1/auth/register", json=auth_data)
    assert response.status_code == 201
    assert response.json()["data"]["email"] == TEST_EMAIL

def test_register_conflict(mock_db, auth_data):
    """Test user registration with existing email."""
    mock_db.execute.side_effect = Exception("already registered")
    
    response = client.post("/api/v1/auth/register", json=auth_data)
    assert response.status_code == 409
    assert "already registered" in response.json()["message"]

def test_login_success(mock_db, auth_data):
    """Test user login."""
    # Mock authenticate_user_native
    mock_row = (uuid4(), TEST_EMAIL, "free", True)
    mock_db.execute.return_value = MagicMock(fetchone=lambda: mock_row)
    
    response = client.post(
        "/api/v1/auth/login", json={"email": TEST_EMAIL, "password": TEST_PASSWORD}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()["data"]

def test_login_failure(mock_db):
    """Test user login failure."""
    mock_db.execute.return_value = MagicMock(fetchone=lambda: None)
    
    response = client.post(
        "/api/v1/auth/login", json={"email": TEST_EMAIL, "password": "wrong_password"}
    )
    assert response.status_code == 401

def test_get_me():
    """Test getting current user info."""
    from src.auth.auth import get_current_active_user
    mock_user = User(
        id=uuid4(),
        email=TEST_EMAIL,
        full_name=TEST_NAME,
        tier="pro",
        is_active=True,
        is_verified=True,
        mfa_enabled=False,
        created_at=datetime.now(UTC)
    )
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 200
    assert response.json()["data"]["email"] == TEST_EMAIL
    
    app.dependency_overrides.pop(get_current_active_user)

def test_refresh_token_success():
    from src.auth.auth import auth_service
    from src.auth.core.tokens import TokenData, TokenPair
    
    # Mock token validation and creation
    mock_token_data = TokenData(
        user_id=str(uuid4()),
        email=TEST_EMAIL,
        tier="pro",
        token_type="refresh",
        exp=datetime.now(UTC),
        iat=datetime.now(UTC),
        jti="test-jti"
    )
    mock_token_pair = TokenPair(
        access_token="new-access",
        refresh_token="new-refresh",
        expires_in=3600
    )
    
    with patch.object(auth_service, "decode_token", return_value=mock_token_data), \
         patch.object(auth_service, "create_token_pair", return_value=mock_token_pair), \
         patch.object(auth_service.token_blacklist, "contains", return_value=False), \
         patch.object(auth_service.token_blacklist, "add", return_value=None):
        
        response = client.post(
            "/api/v1/auth/refresh", json={"refresh_token": "valid-refresh-token"}
        )
        assert response.status_code == 200
        assert response.json()["data"]["access_token"] == "new-access"

def test_logout_success():
    from src.auth.auth import auth_service
    from src.auth.core.tokens import TokenData
    from datetime import datetime, UTC, timedelta
    
    mock_claims = TokenData(
        user_id=str(uuid4()),
        email=TEST_EMAIL,
        tier="pro",
        token_type="access",
        exp=datetime.now(UTC) + timedelta(hours=1),
        iat=datetime.now(UTC),
        jti="test-jti"
    )
    
    with patch("api.middleware.jwt_validator.token_service.decode_token", return_value=mock_claims), \
         patch("api.middleware.jwt_validator.session_service.get_cached_session", new_callable=AsyncMock, return_value=None), \
         patch.object(auth_service, "revoke_token", new_callable=AsyncMock) as mock_revoke:
        
        response = client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": "Bearer some-token"}
        )
        assert response.status_code == 200
        mock_revoke.assert_called_once_with("some-token")

def test_mfa_setup_success(mock_db):
    from src.auth.auth import auth_service
    mock_user = User(id=uuid4(), email=TEST_EMAIL, tier="pro", is_active=True)
    from src.auth.auth import get_current_active_user
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    
    with patch.object(auth_service, "generate_mfa_secret", return_value="ABCDEF"), \
         patch.object(auth_service, "encrypt_mfa_secret", return_value=b"encrypted"), \
         patch.object(auth_service, "get_totp_uri", return_value="otpauth://..."):
        
        response = client.post("/api/v1/auth/mfa/setup")
        assert response.status_code == 200
        assert response.json()["data"]["secret"] == "ABCDEF"
        assert mock_user.mfa_secret == b"encrypted"
    
    app.dependency_overrides.pop(get_current_active_user)
