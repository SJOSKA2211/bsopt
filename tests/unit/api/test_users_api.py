import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from unittest.mock import patch, MagicMock
import os
import uuid
from datetime import datetime, timezone
from src.database.models import User, APIKey
from src.security.auth import get_current_active_user, require_tier
from src.database import get_db

client = TestClient(app)

@pytest.fixture(autouse=True)
def set_testing_env():
    with patch.dict(os.environ, {"TESTING": "true"}):
        yield

@pytest.fixture
def mock_user():
    user = MagicMock(spec=User)
    user.id = str(uuid.uuid4())
    user.email = "test@example.com"
    user.full_name = "Test User"
    user.tier = "free"
    user.is_active = True
    user.is_verified = True
    user.is_mfa_enabled = False
    user.created_at = datetime.now(timezone.utc)
    user.last_login = datetime.now(timezone.utc)
    return user

@pytest.fixture
def mock_db():
    session = MagicMock()
    # Robust mock for query chain
    mock_query = MagicMock()
    session.query.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.count.return_value = 0
    mock_query.offset.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.all.return_value = []
    mock_query.first.return_value = None
    yield session

@pytest.fixture(autouse=True)
def override_dependencies(mock_user, mock_db):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    app.dependency_overrides[get_db] = lambda: mock_db
    yield
    app.dependency_overrides.clear()

def test_get_current_user_profile(mock_user):
    response = client.get("/api/v1/users/me")
    assert response.status_code == 200
    assert response.json()["data"]["email"] == mock_user.email

def test_get_user_stats(mock_user):
    response = client.get("/api/v1/users/me/stats")
    assert response.status_code == 200
    assert "total_requests" in response.json()["data"]

def test_list_api_keys(mock_db, mock_user):
    mock_key = MagicMock(spec=APIKey)
    mock_key.id = uuid.uuid4()
    mock_key.name = "Test Key"
    mock_key.prefix = "bs_123"
    mock_key.created_at = datetime.now(timezone.utc)
    mock_key.last_used_at = datetime.now(timezone.utc)
    
    mock_db.query.return_value.filter.return_value.all.return_value = [mock_key]
    
    response = client.get("/api/v1/users/me/keys")
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1
    assert response.json()["data"][0]["name"] == "Test Key"

def test_create_api_key(mock_db, mock_user):
    def mock_refresh(obj):
        obj.id = uuid.uuid4()
        obj.created_at = datetime.now(timezone.utc)
    mock_db.refresh.side_effect = mock_refresh
    
    response = client.post("/api/v1/users/me/keys", json={"name": "New Key"})
    assert response.status_code == 200
    assert response.json()["data"]["name"] == "New Key"
    assert "raw_key" in response.json()["data"]

def test_revoke_api_key_success(mock_db, mock_user):
    mock_key = MagicMock(spec=APIKey)
    mock_db.query.return_value.filter.return_value.first.return_value = mock_key
    
    response = client.delete(f"/api/v1/users/me/keys/{uuid.uuid4()}")
    assert response.status_code == 200
    assert mock_key.is_active is False

def test_revoke_api_key_not_found(mock_db, mock_user):
    mock_db.query.return_value.filter.return_value.first.return_value = None
    
    response = client.delete(f"/api/v1/users/me/keys/{uuid.uuid4()}")
    assert response.status_code == 404

def test_update_current_user_profile_success(mock_db, mock_user):
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    
    with patch('src.api.routes.users.log_audit'):
        response = client.patch("/api/v1/users/me", json={"full_name": "Updated Name"})
        assert response.status_code == 200
        assert mock_user.full_name == "Updated Name"

def test_update_current_user_profile_email_conflict(mock_db, mock_user):
    # First call to first() gets db_user, second call (to check existing email) gets another user
    mock_db.query.return_value.filter.return_value.first.side_effect = [mock_user, MagicMock()]
    
    response = client.patch("/api/v1/users/me", json={"email": "conflict@example.com"})
    assert response.status_code == 409

def test_update_current_user_profile_email_success(mock_db, mock_user):
    # First call gets db_user, second call (check email) gets None
    mock_db.query.return_value.filter.return_value.first.side_effect = [mock_user, None]
    
    with patch('src.api.routes.users.log_audit'):
        response = client.patch("/api/v1/users/me", json={"email": "new@example.com"})
        assert response.status_code == 200
        assert mock_user.email == "new@example.com"
        assert mock_user.is_verified is False
        mock_db.commit.assert_called_once()

def test_update_current_user_profile_no_user(mock_db, mock_user):
    mock_db.query.return_value.filter.return_value.first.return_value = None
    response = client.patch("/api/v1/users/me", json={"full_name": "New Name"})
    assert response.status_code == 404

def test_delete_current_user_account_success(mock_db, mock_user):
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    
    with patch('src.api.routes.users.log_audit'):
        response = client.delete("/api/v1/users/me")
        assert response.status_code == 200
        assert mock_user.is_active is False

def test_delete_current_user_account_failure(mock_db, mock_user):
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    mock_db.commit.side_effect = Exception("DB Error")
    
    response = client.delete("/api/v1/users/me")
    assert response.status_code == 500

def test_get_user_by_id_enterprise(mock_db, mock_user):
    mock_user.tier = "enterprise"
    
    mock_target_user = MagicMock(spec=User)
    mock_target_user.id = str(uuid.uuid4())
    mock_target_user.email = "target@example.com"
    mock_target_user.full_name = "Target User"
    mock_target_user.tier = "free"
    mock_target_user.is_active = True
    mock_target_user.is_verified = True
    mock_target_user.is_mfa_enabled = False
    mock_target_user.created_at = datetime.now(timezone.utc)
    mock_target_user.last_login = datetime.now(timezone.utc)
    
    mock_db.query.return_value.filter.return_value.first.return_value = mock_target_user
    
    response = client.get(f"/api/v1/users/{mock_target_user.id}")
    assert response.status_code == 200
    assert response.json()["data"]["email"] == "target@example.com"

def test_get_user_by_id_not_found(mock_db, mock_user):
    mock_user.tier = "enterprise"
    mock_db.query.return_value.filter.return_value.first.return_value = None
    
    response = client.get(f"/api/v1/users/{uuid.uuid4()}")
    assert response.status_code == 404

def test_list_users_enterprise(mock_db, mock_user):
    mock_user.tier = "enterprise"
    
    mock_query = mock_db.query.return_value
    mock_query.count.return_value = 1
    
    mock_user_res = MagicMock(spec=User)
    mock_user_res.id = str(uuid.uuid4())
    mock_user_res.email = "res@example.com"
    mock_user_res.full_name = "Res User"
    mock_user_res.tier = "free"
    mock_user_res.is_active = True
    mock_user_res.is_verified = True
    mock_user_res.is_mfa_enabled = False
    mock_user_res.created_at = datetime.now(timezone.utc)
    mock_user_res.last_login = datetime.now(timezone.utc)
    
    mock_query.all.return_value = [mock_user_res]
    
    response = client.get("/api/v1/users?search=test&page=2")
    assert response.status_code == 200
    assert response.json()["pagination"]["total"] == 1
    assert response.json()["pagination"]["has_prev"] is True