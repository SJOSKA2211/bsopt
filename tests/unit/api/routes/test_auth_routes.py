import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI, BackgroundTasks
from src.api.routes.auth import router
from src.database.models import User
from src.security.auth import get_current_user, get_current_active_user, get_auth_service
from src.security.password import get_password_service
from src.database import get_db
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock

app = FastAPI()
app.include_router(router)

# Mock dependencies
def get_mock_user():
    return User(id=uuid.uuid4(), email="test@example.com", tier="free", is_active=True, is_verified=True)

app.dependency_overrides[get_current_user] = get_mock_user
app.dependency_overrides[get_current_active_user] = get_mock_user

client = TestClient(app)

@pytest.fixture
def mock_auth_service():
    mock = MagicMock()
    app.dependency_overrides[get_auth_service] = lambda: mock
    yield mock
    app.dependency_overrides.pop(get_auth_service, None)

@pytest.fixture
def mock_password_service():
    mock = MagicMock()
    app.dependency_overrides[get_password_service] = lambda: mock
    yield mock
    app.dependency_overrides.pop(get_password_service, None)

@pytest.fixture
def mock_db():
    mock = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock
    yield mock
    app.dependency_overrides.pop(get_db, None)

def test_login_success(mock_auth_service, mock_db):
    mock_user = get_mock_user()
    mock_auth_service.authenticate_user.return_value = mock_user
    mock_token_pair = MagicMock()
    mock_token_pair.access_token = "access"
    mock_token_pair.refresh_token = "refresh"
    mock_token_pair.token_type = "bearer"
    mock_token_pair.expires_in = 3600
    mock_auth_service.create_token_pair.return_value = mock_token_pair
    
    response = client.post("/auth/login", json={"email": "test@example.com", "password": "Password123!"})
    assert response.status_code == 200
    assert response.json()["data"]["access_token"] == "access"

def test_login_mfa_required(mock_auth_service, mock_db):
    mock_user = get_mock_user()
    mock_user.is_mfa_enabled = True
    mock_auth_service.authenticate_user.return_value = mock_user
    
    response = client.post("/auth/login", json={"email": "test@example.com", "password": "Password123!"})
    assert response.status_code == 200
    assert response.json()["data"]["requires_mfa"] is True

def test_login_mfa_verify(mock_auth_service, mock_db, mocker):
    mock_user = get_mock_user()
    mock_user.is_mfa_enabled = True
    mock_user.mfa_secret = "secret"
    mock_auth_service.authenticate_user.return_value = mock_user
    mock_token_pair = MagicMock()
    mock_token_pair.access_token = "access"
    mock_token_pair.refresh_token = "refresh"
    mock_token_pair.token_type = "bearer"
    mock_token_pair.expires_in = 3600
    mock_auth_service.create_token_pair.return_value = mock_token_pair
    
    mocker.patch("src.api.routes.auth._verify_mfa_code", return_value=True)
    
    response = client.post("/auth/login", json={"email": "test@example.com", "password": "Password123!", "mfa_code": "123456"})
    assert response.status_code == 200
    assert response.json()["data"]["requires_mfa"] is False

def test_logout(mock_auth_service):
    mock_auth_service.invalidate_token = AsyncMock()
    response = client.post("/auth/logout", headers={"Authorization": "Bearer some-token"})
    assert response.status_code == 200
    assert mock_auth_service.invalidate_token.called

def test_register_success(mock_password_service, mock_db, mocker):
    mock_db.query.return_value.filter.return_value.first.return_value = None
    
    mock_password_service.validate_password.return_value.is_valid = True
    mock_password_service.hash_password.return_value = "hashed"
    mock_password_service.generate_verification_token.return_value = "token"
    mocker.patch("src.api.routes.auth.idempotency_manager.check_and_set", return_value=True)
    mocker.patch("src.api.routes.auth._send_verification_email")
    
    # Mock created user after refresh
    def mock_refresh(obj):
        obj.id = uuid.uuid4()
        obj.email = "new@example.com"
    mock_db.refresh.side_effect = mock_refresh
    
    response = client.post("/auth/register", json={
        "email": "new@example.com",
        "password": "Password123!",
        "password_confirm": "Password123!",
        "accept_terms": True,
        "full_name": "New User"
    })
    assert response.status_code == 201
    assert response.json()["data"]["email"] == "new@example.com"

def test_verify_email_success(mock_db):
    mock_user = get_mock_user()
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    
    response = client.post("/auth/verify-email", json={"token": "valid-token"})
    assert response.status_code == 200
    assert mock_user.is_verified is True

def test_refresh_token_success(mock_auth_service, mock_db):
    mock_user = get_mock_user()
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    
    from src.security.auth import TokenData
    mock_auth_service.validate_token = AsyncMock(return_value=TokenData(
        user_id=str(mock_user.id),
        email=mock_user.email,
        tier=mock_user.tier,
        token_type="refresh",
        exp=datetime.now(timezone.utc),
        iat=datetime.now(timezone.utc)
    ))
    mock_auth_service.invalidate_token = AsyncMock()
    mock_token_pair = MagicMock()
    mock_token_pair.access_token = "new-access"
    mock_token_pair.refresh_token = "new-refresh"
    mock_token_pair.token_type = "bearer"
    mock_token_pair.expires_in = 3600
    mock_auth_service.create_token_pair.return_value = mock_token_pair
    
    response = client.post("/auth/refresh", json={"refresh_token": "old-refresh"})
    assert response.status_code == 200
    assert response.json()["data"]["access_token"] == "new-access"