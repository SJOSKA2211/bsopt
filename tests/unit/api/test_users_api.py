import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from src.api.main import app
from src.database.models import User
from src.database import get_db
from src.security.auth import get_current_active_user, require_tier
import uuid
from datetime import datetime, timezone
from src.api.exceptions import AuthenticationException

client = TestClient(app)

# Helper function for unauthorized dependency override
def raise_auth_exception():
    raise AuthenticationException()

@pytest.fixture
def mock_user():
    return User(
        id=uuid.uuid4(),
        email="test@example.com",
        full_name="Test User",
        tier="free",
        is_active=True,
        is_verified=True,
        is_mfa_enabled=False,
        created_at=datetime.now(timezone.utc)
    )

@pytest.fixture
def enterprise_user():
    return User(
        id=uuid.uuid4(),
        email="enterprise@example.com",
        full_name="Enterprise User",
        tier="enterprise",
        is_active=True,
        is_verified=True,
        is_mfa_enabled=False,
        created_at=datetime.now(timezone.utc)
    )

def test_get_me_success(mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    response = client.get("/api/v1/users/me")
    assert response.status_code == 200
    assert response.json()["data"]["email"] == mock_user.email
    app.dependency_overrides = {}

def test_get_me_unauthorized():
    app.dependency_overrides[get_current_active_user] = raise_auth_exception
    response = client.get("/api/v1/users/me")
    assert response.status_code == 401
    app.dependency_overrides = {}

def test_update_me_success(mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    response = client.patch("/api/v1/users/me", json={"full_name": "Updated Name"})
    assert response.status_code == 200
    assert response.json()["data"]["full_name"] == "Updated Name"
    app.dependency_overrides = {}

def test_update_me_email_conflict(mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.side_effect = [mock_user, MagicMock()]
    response = client.patch("/api/v1/users/me", json={"email": "taken@example.com"})
    assert response.status_code == 409
    app.dependency_overrides = {}

def test_update_me_persistence_error(mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    mock_db.commit.side_effect = Exception("DB error")
    response = client.patch("/api/v1/users/me", json={"full_name": "New Name"})
    assert response.status_code == 500
    app.dependency_overrides = {}

def test_delete_me_success(mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    response = client.delete("/api/v1/users/me")
    assert response.status_code == 200
    app.dependency_overrides = {}

def test_delete_me_persistence_error(mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    mock_db.commit.side_effect = Exception("DB error")
    response = client.delete("/api/v1/users/me")
    assert response.status_code == 500
    app.dependency_overrides = {}

def test_get_user_stats(mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    response = client.get("/api/v1/users/me/stats")
    assert response.status_code == 200
    assert "total_requests" in response.json()["data"]
    app.dependency_overrides = {}

def test_get_user_by_id_enterprise(enterprise_user):
    app.dependency_overrides[get_current_active_user] = lambda: enterprise_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = enterprise_user
    response = client.get(f"/api/v1/users/{enterprise_user.id}")
    assert response.status_code == 200
    app.dependency_overrides = {}

def test_get_user_by_id_not_found(enterprise_user):
    app.dependency_overrides[get_current_active_user] = lambda: enterprise_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = None
    response = client.get(f"/api/v1/users/{uuid.uuid4()}")
    assert response.status_code == 404
    app.dependency_overrides = {}

def test_list_users_enterprise(enterprise_user, mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: enterprise_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.count.return_value = 2
    mock_db.query.return_value.offset.return_value.limit.return_value.all.return_value = [enterprise_user, mock_user]
    response = client.get("/api/v1/users")
    assert response.status_code == 200
    assert len(response.json()["items"]) == 2
    app.dependency_overrides = {}

def test_list_users_empty(enterprise_user):
    app.dependency_overrides[get_current_active_user] = lambda: enterprise_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.count.return_value = 0
    mock_db.query.return_value.offset.return_value.limit.return_value.all.return_value = []
    response = client.get("/api/v1/users")
    assert response.status_code == 200
    assert len(response.json()["items"]) == 0
    app.dependency_overrides = {}

def test_list_users_with_search(enterprise_user):
    app.dependency_overrides[get_current_active_user] = lambda: enterprise_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.count.return_value = 1
    mock_db.query.return_value.filter.return_value.offset.return_value.limit.return_value.all.return_value = [enterprise_user]
    response = client.get("/api/v1/users?search=enterprise")
    assert response.status_code == 200
    assert len(response.json()["items"]) == 1
    app.dependency_overrides = {}

def test_get_user_by_id_insufficient_tier(mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    # The require_tier dependency will raise a 403 error, which is handled by our exception handlers.
    response = client.get(f"/api/v1/users/{uuid.uuid4()}")
    assert response.status_code == 403
    app.dependency_overrides = {}

def test_list_users_with_tier_filter(enterprise_user, mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: enterprise_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.count.return_value = 1
    mock_db.query.return_value.filter.return_value.offset.return_value.limit.return_value.all.return_value = [mock_user]
    response = client.get("/api/v1/users?tier=free")
    assert response.status_code == 200
    assert len(response.json()["items"]) == 1
    assert response.json()["items"][0]["tier"] == "free"
    app.dependency_overrides = {}
