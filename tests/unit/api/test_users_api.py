import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.exceptions import AuthenticationException
from src.api.main import app
from src.database import get_db
from src.database.models import User
from src.security.auth import get_current_active_user

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
        created_at=datetime.now(UTC)
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
        created_at=datetime.now(UTC)
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

@patch("src.api.routes.users.publish_to_redis", new_callable=AsyncMock)
def test_update_me_success(mock_publish_to_redis, mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    response = client.patch("/api/v1/users/me", json={"full_name": "Updated Name"})
    assert response.status_code == 200
    assert response.json()["data"]["full_name"] == "Updated Name"
    mock_publish_to_redis.assert_called_once() # Assert Redis publish
    app.dependency_overrides = {}

@patch("src.api.routes.users.publish_to_redis", new_callable=AsyncMock)
def test_update_me_success_email(mock_publish_to_redis, mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.side_effect = [mock_user, None] # First call finds user, second finds no conflict
    response = client.patch("/api/v1/users/me", json={"email": "new_email@example.com"})
    assert response.status_code == 200
    assert response.json()["data"]["email"] == "new_email@example.com"
    mock_publish_to_redis.assert_called_once() # Assert Redis publish
    app.dependency_overrides = {}

def test_update_me_no_changes(mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    # Send request with current email and full_name, expecting no changes
    response = client.patch("/api/v1/users/me", json={"email": mock_user.email, "full_name": mock_user.full_name})
    assert response.status_code == 200
    assert response.json()["data"]["full_name"] == mock_user.full_name
    # Ensure commit was NOT called as no changes were made
    mock_db.commit.assert_not_called()
    app.dependency_overrides = {}

def test_update_me_email_conflict(mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    # First call to filter.first returns mock_user (current user), second returns existing user (conflict)
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

@patch("src.api.routes.users.publish_to_redis", new_callable=AsyncMock)
def test_delete_me_success(mock_publish_to_redis, mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = mock_user
    response = client.delete("/api/v1/users/me")
    assert response.status_code == 200
    mock_user.is_active = False # Verify soft delete
    mock_db.commit.assert_called_once()
    mock_publish_to_redis.assert_called_once() # Assert Redis publish
    app.dependency_overrides = {}

def test_delete_me_not_found(mock_user):
    app.dependency_overrides[get_current_active_user] = lambda: mock_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = None # User not found
    response = client.delete("/api/v1/users/me")
    assert response.status_code == 404
    mock_db.commit.assert_not_called()
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
    # Chain .filter().count() and .filter().offset().limit().all()
    mock_query = mock_db.query.return_value
    mock_filter = mock_query.filter.return_value
    mock_filter.count.return_value = 1
    mock_filter.offset.return_value.limit.return_value.all.return_value = [enterprise_user]
    
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

    mock_query = mock_db.query.return_value

    mock_filter = mock_query.filter.return_value

    mock_filter.count.return_value = 1

    mock_filter.offset.return_value.limit.return_value.all.return_value = [mock_user]

    

    response = client.get("/api/v1/users?tier=free")

    assert response.status_code == 200

    assert len(response.json()["items"]) == 1

    assert response.json()["items"][0]["tier"] == "free"

    app.dependency_overrides = {}



def test_list_users_with_is_active_filter(enterprise_user):

    app.dependency_overrides[get_current_active_user] = lambda: enterprise_user

    mock_db = MagicMock()

    app.dependency_overrides[get_db] = lambda: mock_db

    

    active_user = User(

        id=uuid.uuid4(),

        email="active@example.com",

        full_name="Active User",

        tier="free",

        is_active=True,

        is_verified=True,

        is_mfa_enabled=False,

        created_at=datetime.now(UTC)

    )

    

    mock_query = mock_db.query.return_value

    # Use side_effect to return the SAME mock_filter for consecutive .filter() calls if needed, 

    # but here we just need it once per request.

    mock_filter = MagicMock()

    mock_query.filter.return_value = mock_filter

    mock_filter.count.return_value = 1

    mock_filter.offset.return_value.limit.return_value.all.return_value = [active_user]

    

    response = client.get("/api/v1/users?is_active=true")

    assert response.status_code == 200

    assert len(response.json()["items"]) == 1

    assert response.json()["items"][0]["is_active"] is True

    app.dependency_overrides = {}
