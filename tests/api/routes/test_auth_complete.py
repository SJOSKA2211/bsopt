import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.database.models import User
from datetime import datetime, timezone, timedelta
import uuid

# Re-use fixtures from conftest
@pytest.fixture
def mock_background_tasks():
    with patch("src.api.routes.auth.BackgroundTasks") as mock:
        yield mock

@pytest.fixture
def mock_send_email():
    with patch("src.api.routes.auth._send_verification_email") as mock:
        yield mock

@pytest.mark.asyncio
async def test_register_flow(api_client, mock_db_session, mock_send_email):
    # 1. Register Success
    # Use a random password to avoid policy issues (e.g. breached passwords) if configured
    password = f"Secure_{uuid.uuid4()}!_Pass"
    payload = {
        "email": "newuser@example.com",
        "password": password,
        "password_confirm": password,
        "full_name": "New User",
        "accept_terms": True
    }
    response = api_client.post("/api/v1/auth/register", json=payload)
    if response.status_code != 201:
        print(response.json()) # For debugging if it fails
    assert response.status_code == 201
    data = response.json()["data"]
    assert data["email"] == payload["email"]
    assert data["verification_required"] is True
    
    # Verify DB state
    user = mock_db_session.query(User).filter(User.email == payload["email"]).first()
    assert user is not None
    assert user.verification_token is not None
    
    # 2. Register Duplicate
    response = api_client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 409

@pytest.mark.asyncio
async def test_verify_email(api_client, mock_db_session):
    # Setup user with token
    token = "testtoken123"
    user = User(
        email="verify@example.com",
        hashed_password="hash",
        is_verified=False,
        verification_token=f"verify:{token}"
    )
    mock_db_session.add(user)
    
    # We need to manually patch the query because mock_db_session only supports id/email lookup
    # The verify_email route queries by verification_token
    
    # Override query side effect for this test to return our user when filter is called
    original_query = mock_db_session.query
    
    # Mocking query chain: db.query(User).filter(...).first()
    # We create a mock query object that returns our user on first()
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.first.return_value = user
    
    # We patch query to return our mock_query ONLY if model is User
    # But mock_db_session.query is a MagicMock with side_effect.
    # We can replace side_effect.
    
    def side_effect(model):
        if model == User:
            return mock_query
        return original_query(model) # Fallback? original_query is the mock's previous side_effect wrapper?
        # Actually original_query was a bound method of MagicMock or side_effect function?
        # It was a lambda in conftest.
    
    # Let's just set return_value of query(User) if possible, but query is called with arg.
    mock_db_session.query.side_effect = side_effect
    
    # 1. Verify Success
    response = api_client.post("/api/v1/auth/verify-email", json={"token": token})
    if response.status_code != 200:
         print(response.json())
    assert response.status_code == 200
    assert "verified successfully" in response.json()["message"]
    assert user.is_verified is True
    assert user.verification_token is None
    
    # 2. Verify Invalid Token
    # Force first() to return None
    mock_query.first.return_value = None
    response = api_client.post("/api/v1/auth/verify-email", json={"token": "wrong"})
    assert response.status_code in [400, 422]

@pytest.mark.asyncio
async def test_login_flow(api_client, mock_db_session):
    # Setup
    password = f"Secure_{uuid.uuid4()}!_Pass"
    
    with patch("src.security.password.PasswordService.verify_password", return_value=True):
        user = User(
            email="login@example.com",
            hashed_password="hashed_secret",
            is_verified=True,
            is_active=True,
            is_mfa_enabled=False
        )
        mock_db_session.add(user)
        
        # 1. Login Success
        response = api_client.post("/api/v1/auth/login", json={
            "email": "login@example.com",
            "password": password
        })
        assert response.status_code == 200
        assert "access_token" in response.json()["data"]
        
        # 2. Login Unverified
        user.is_verified = False
        response = api_client.post("/api/v1/auth/login", json={
            "email": "login@example.com",
            "password": password
        })
        assert response.status_code == 403 # PermissionDenied
        
        # 3. Login Inactive
        user.is_verified = True
        user.is_active = False
        response = api_client.post("/api/v1/auth/login", json={
            "email": "login@example.com",
            "password": password
        })
        # Authenticate user returns None for inactive user, leading to 401
        assert response.status_code == 401

@pytest.mark.asyncio
async def test_login_mfa(api_client, mock_db_session):
    # Setup
    with patch("src.security.password.PasswordService.verify_password", return_value=True):
        user = User(
            email="mfa@example.com",
            hashed_password="hash",
            is_verified=True,
            is_active=True,
            is_mfa_enabled=True,
            mfa_secret="secret"
        )
        mock_db_session.add(user)
        
        # 1. Login without code -> MFA Required
        response = api_client.post("/api/v1/auth/login", json={
            "email": "mfa@example.com",
            "password": "pass"
        })
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["requires_mfa"] is True
        assert data["access_token"] == ""
        
        # 2. Login with Invalid Code
        with patch("src.api.routes.auth._verify_mfa_code", return_value=False):
            response = api_client.post("/api/v1/auth/login", json={
                "email": "mfa@example.com",
                "password": "pass",
                "mfa_code": "000000"
            })
            assert response.status_code == 401 # AuthenticationException
            
        # 3. Login with Valid Code
        with patch("src.api.routes.auth._verify_mfa_code", return_value=True):
            response = api_client.post("/api/v1/auth/login", json={
                "email": "mfa@example.com",
                "password": "pass",
                "mfa_code": "123456"
            })
            assert response.status_code == 200
            assert "access_token" in response.json()["data"]
            assert response.json()["data"]["access_token"] != ""

@pytest.mark.asyncio
async def test_refresh_token(api_client, mock_db_session):
    from src.security.auth import TokenData
    
    user_id = uuid.uuid4()
    user = User(id=user_id, email="test@example.com", is_active=True, tier="free")
    mock_db_session.add(user)
    
    # We need to mock validate_token with AsyncMock because it is awaited
    with patch("src.security.auth.AuthService.validate_token", new_callable=AsyncMock) as mock_validate:
        # Mock return value of validate_token. Use real TokenData object.
        now = datetime.now(timezone.utc)
        mock_validate.return_value = TokenData(
            user_id=str(user_id),
            email="test@example.com",
            tier="free",
            token_type="refresh",
            exp=now + timedelta(days=1),
            iat=now
        )
        
        # We also need mock create_token_pair
        with patch("src.security.auth.AuthService.create_token_pair") as mock_create:
            mock_create.return_value = MagicMock(
                access_token="new_access",
                refresh_token="new_refresh",
                token_type="bearer",
                expires_in=3600
            )
            
            response = api_client.post("/api/v1/auth/refresh", json={"refresh_token": "valid_refresh"})
            if response.status_code != 200:
                print(response.json())
            assert response.status_code == 200
            assert response.json()["data"]["access_token"] == "new_access"
@pytest.mark.asyncio
async def test_logout(api_client):
    # Mock invalidate_token. Correct path: src.security.auth.AuthService
    with patch("src.security.auth.AuthService.invalidate_token", new_callable=AsyncMock) as mock_invalidate:
        # We need a valid user dependency for logout
        from src.security.auth import get_current_user
        app = api_client.app
        app.dependency_overrides[get_current_user] = lambda: User(id=uuid.uuid4(), email="test@example.com")
        
        response = api_client.post("/api/v1/auth/logout", headers={"Authorization": "Bearer token"})
        assert response.status_code == 200
        mock_invalidate.assert_called_once()