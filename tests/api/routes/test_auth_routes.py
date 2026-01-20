import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.database import get_db
from src.security.auth import get_auth_service
from src.security.password import get_password_service
from src.database.models import User
from unittest.mock import MagicMock, patch, AsyncMock
import uuid

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
    mock_session = MagicMock()
    # Mock common session methods to return mocks that don't trigger real DB
    mock_session.query.return_value.filter.return_value.first.return_value = None
    app.dependency_overrides[get_db] = lambda: mock_session
    yield mock_session
    app.dependency_overrides.pop(get_db, None)

def test_login_success(mock_auth_service, mock_db):
    mock_user = MagicMock(spec=User)
    mock_user.id = uuid.uuid4()
    mock_user.email = "test@ex.com"
    mock_user.is_verified = True
    mock_user.is_active = True
    mock_user.is_mfa_enabled = False
    mock_user.tier = "free"
    
    mock_auth_service.authenticate_user.return_value = mock_user
    mock_token_pair = MagicMock()
    mock_token_pair.access_token = "at"
    mock_token_pair.refresh_token = "rt"
    mock_token_pair.token_type = "bearer"
    mock_token_pair.expires_in = 3600
    mock_auth_service.create_token_pair.return_value = mock_token_pair
    
    payload = {"email": "test@ex.com", "password": "password"}
    response = client.post("/api/v1/auth/login", json=payload)
    
    assert response.status_code == 200
    assert response.json()["data"]["access_token"] == "at"

def test_register_success(mock_password_service, mock_db):
    mock_db.query.return_value.filter.return_value.first.return_value = None
    
    mock_val = MagicMock()
    mock_val.is_valid = True
    mock_password_service.validate_password.return_value = mock_val
    mock_password_service.hash_password.return_value = "hashed"
    mock_password_service.generate_verification_token.return_value = "token"
    
    payload = {
        "email": "new@ex.com",
        "password": "SecurePassword123!",
        "password_confirm": "SecurePassword123!",
        "full_name": "New User",
        "accept_terms": True
    }
    
    with patch("src.api.routes.auth._send_verification_email") as mock_send:
        response = client.post("/api/v1/auth/register", json=payload)
        assert response.status_code == 201
        mock_db.add.assert_called()

def test_verify_email_success(mock_db):
    mock_user = MagicMock(spec=User)
    # Patch first() to return the mock user directly
    with patch.object(mock_db.query.return_value.filter.return_value, "first", return_value=mock_user):
        payload = {"token": "valid_token"}
        response = client.post("/api/v1/auth/verify-email", json=payload)
        
        assert response.status_code == 200
        assert mock_user.is_verified is True

def test_refresh_token_success(mock_auth_service, mock_db):
    mock_token_data = MagicMock()
    mock_token_data.token_type = "refresh"
    mock_token_data.user_id = str(uuid.uuid4())
    mock_auth_service.validate_token = AsyncMock(return_value=mock_token_data)
    mock_auth_service.invalidate_token = AsyncMock()
    
    mock_user = MagicMock(spec=User)
    mock_user.is_active = True
    mock_user.email = "test@ex.com"
    mock_user.tier = "free"
    
    with patch.object(mock_db.query.return_value.filter.return_value, "first", return_value=mock_user):
        mock_token_pair = MagicMock()
        mock_token_pair.access_token = "new_at"
        mock_token_pair.refresh_token = "new_rt"
        mock_token_pair.token_type = "bearer"
        mock_token_pair.expires_in = 3600
        mock_auth_service.create_token_pair.return_value = mock_token_pair
        
        payload = {"refresh_token": "old_rt"}
        response = client.post("/api/v1/auth/refresh", json=payload)
        
        assert response.status_code == 200
        assert response.json()["data"]["access_token"] == "new_at"
