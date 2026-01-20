import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.api.main import app
from src.database.models import User, APIKey
from src.security.auth import get_current_active_user
from src.database import get_db
import uuid
from datetime import datetime, timezone

client = TestClient(app)

@pytest.fixture
def mock_user():
    user = User(
        id=uuid.uuid4(),
        email="test@example.com",
        full_name="Test User",
        tier="enterprise",
        is_active=True,
        is_verified=True,
        is_mfa_enabled=False,
        created_at=datetime.now(timezone.utc),
        last_login=datetime.now(timezone.utc)
    )
    return user

@pytest.fixture(autouse=True)
def override_user_deps(mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    app.dependency_overrides[get_db] = lambda: MagicMock()
    yield
    app.dependency_overrides = {}

def test_get_me(mock_user):
    response = client.get("/api/v1/users/me")
    assert response.status_code == 200
    assert response.json()["data"]["email"] == "test@example.com"

def test_update_me(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    
    payload = {"full_name": "Updated Name"}
    
    with patch("src.api.routes.users.log_audit") as mock_audit:
        response = client.patch("/api/v1/users/me", json=payload)
        assert response.status_code == 200
        assert mock_user.full_name == "Updated Name"

def test_update_me_email_conflict(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.side_effect = [mock_user, User(email="other@ex.com")]
    
    payload = {"email": "other@ex.com"}
    response = client.patch("/api/v1/users/me", json=payload)
    assert response.status_code == 409

def test_get_user_stats(mock_user):
    response = client.get("/api/v1/users/me/stats")
    assert response.status_code == 200
    assert "total_requests" in response.json()["data"]

def test_list_api_keys(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_key = APIKey(
        id=uuid.uuid4(), 
        name="Key 1", 
        prefix="bs_abc", 
        created_at=datetime.now(timezone.utc), 
        last_used_at=None
    )
    mock_db.query.return_value.filter.return_value.all.return_value = [mock_key]
    
    response = client.get("/api/v1/users/me/keys")
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1

def test_create_api_key(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    
    def mock_refresh(obj):
        obj.created_at = datetime.now(timezone.utc)
        obj.id = uuid.uuid4()
    
    mock_db.refresh.side_effect = mock_refresh
    
    payload = {"name": "New Key"}
    response = client.post("/api/v1/users/me/keys", json=payload)
    assert response.status_code == 200
    assert "raw_key" in response.json()["data"]
    assert response.json()["data"]["name"] == "New Key"

def test_revoke_api_key(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_key = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = mock_key
    
    response = client.delete(f"/api/v1/users/me/keys/{uuid.uuid4()}")
    assert response.status_code == 200
    assert mock_key.is_active is False

def test_delete_me(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    
    response = client.delete("/api/v1/users/me")
    assert response.status_code == 200
    assert mock_user.is_active is False

def test_enterprise_get_user(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    
    user_id = str(uuid.uuid4())
    response = client.get(f"/api/v1/users/{user_id}")
    assert response.status_code == 200

def test_enterprise_list_users(mock_user):
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    # Chain: db.query(User).offset(...).limit(...).all()
    mock_query = mock_db.query.return_value
    mock_query.offset.return_value.limit.return_value.all.return_value = [mock_user]
    mock_query.count.return_value = 1
    
    response = client.get("/api/v1/users")
    assert response.status_code == 200
    assert len(response.json()["items"]) == 1
