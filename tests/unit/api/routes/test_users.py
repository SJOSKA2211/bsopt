import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from src.api.routes.users import router
from src.database.models import User, APIKey
from src.security.auth import get_current_active_user, require_tier
from src.database import get_db
import uuid
from datetime import datetime, timezone

app = FastAPI()
app.include_router(router)

# Mock dependencies
def get_mock_user():
    return User(
        id=uuid.uuid4(), 
        email="test@example.com", 
        full_name="Test User",
        tier="free", 
        is_active=True,
        is_verified=True,
        is_mfa_enabled=False,
        created_at=datetime.now(timezone.utc),
        last_login=datetime.now(timezone.utc)
    )

app.dependency_overrides[get_current_active_user] = get_mock_user
app.dependency_overrides[get_db] = lambda: None # Will mock session individually

# Setup for tier requirements
from src.security.auth import require_tier
def override_require_tier(allowed_tiers):
    async def _override():
        return None
    return _override

client = TestClient(app)

def test_list_api_keys(mocker):
    mock_db = mocker.Mock()
    app.dependency_overrides[get_db] = lambda: mock_db
    
    mock_key = APIKey(
        id=uuid.uuid4(), 
        name="test", 
        prefix="bs_abc", 
        created_at=datetime.now(timezone.utc),
        is_active=True
    )
    mock_db.query.return_value.filter.return_value.all.return_value = [mock_key]
    
    response = client.get("/users/me/keys")
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1

def test_create_api_key(mocker):
    mock_db = mocker.Mock()
    app.dependency_overrides[get_db] = lambda: mock_db
    
    # Mock return value of refresh or just ensure it has fields
    def mock_refresh(obj):
        obj.id = uuid.uuid4()
        obj.created_at = datetime.now(timezone.utc)
    mock_db.refresh.side_effect = mock_refresh
    
    response = client.post("/users/me/keys", json={"name": "new_key"})
    assert response.status_code == 200
    assert "raw_key" in response.json()["data"]
    assert mock_db.add.called

def test_revoke_api_key_success(mocker):
    mock_db = mocker.Mock()
    app.dependency_overrides[get_db] = lambda: mock_db
    
    mock_key = APIKey(id=uuid.uuid4(), is_active=True)
    mock_db.query.return_value.filter.return_value.first.return_value = mock_key
    
    response = client.delete(f"/users/me/keys/{mock_key.id}")
    assert response.status_code == 200
    assert mock_key.is_active is False

def test_revoke_api_key_not_found(mocker):
    mock_db = mocker.Mock()
    app.dependency_overrides[get_db] = lambda: mock_db
    mock_db.query.return_value.filter.return_value.first.return_value = None
    
    response = client.delete("/users/me/keys/invalid")
    assert response.status_code == 404

def test_get_me():
    response = client.get("/users/me")
    assert response.status_code == 200
    assert response.json()["data"]["email"] == "test@example.com"

def test_update_me_success(mocker):
    mock_db = mocker.Mock()
    app.dependency_overrides[get_db] = lambda: mock_db
    
    user = get_mock_user()
    mock_db.query.return_value.filter.return_value.first.side_effect = [user, None] # User exists, Email not taken
    
    # Mock UserResponse.from_orm to return valid data
    from src.api.schemas.user import UserResponse
    mocker.patch("src.api.schemas.user.UserResponse.from_orm", return_value=UserResponse(
        id=user.id, email=user.email, full_name="New Name", tier="free", 
        is_active=True, is_verified=True, is_mfa_enabled=False, created_at=datetime.now()
    ))
    
    response = client.patch("/users/me", json={"full_name": "New Name", "email": "new@example.com"})
    assert response.status_code == 200
    assert user.full_name == "New Name"

def test_update_me_conflict(mocker):
    mock_db = mocker.Mock()
    app.dependency_overrides[get_db] = lambda: mock_db
    
    user = get_mock_user()
    mock_db.query.return_value.filter.return_value.first.side_effect = [user, User()] # User exists, Email taken
    
    response = client.patch("/users/me", json={"email": "taken@example.com"})
    assert response.status_code == 409

def test_get_stats():
    response = client.get("/users/me/stats")
    assert response.status_code == 200
    assert "total_requests" in response.json()["data"]

def test_delete_me_success(mocker):
    mock_db = mocker.Mock()
    app.dependency_overrides[get_db] = lambda: mock_db
    user = get_mock_user()
    mock_db.query.return_value.filter.return_value.first.return_value = user
    
    response = client.delete("/users/me")
    assert response.status_code == 200
    assert user.is_active is False

def test_delete_me_fail(mocker):
    mock_db = mocker.Mock()
    app.dependency_overrides[get_db] = lambda: mock_db
    user = get_mock_user()
    mock_db.query.return_value.filter.return_value.first.return_value = user
    mock_db.commit.side_effect = Exception("DB error")
    
    response = client.delete("/users/me")
    assert response.status_code == 500

def test_get_user_by_id_enterprise(mocker):
    mock_db = mocker.Mock()
    app.dependency_overrides[get_db] = lambda: mock_db
    user = get_mock_user()
    mock_db.query.return_value.filter.return_value.first.return_value = user
    
    # We need to find the exact dependency instance used in the route
    # Or just mock the entire auth check in auth.py if possible
    # A cleaner way is to mock get_current_active_user to return an enterprise user
    def get_enterprise_user():
        u = get_mock_user()
        u.tier = "enterprise"
        return u
    app.dependency_overrides[get_current_active_user] = get_enterprise_user
    
    # Also need to override require_tier(["enterprise"])
    # We can use a trick to override all dependencies that match a pattern or just the one
    for route in app.routes:
        if route.path == "/users/{user_id}":
            for dep in route.dependencies:
                app.dependency_overrides[dep.dependency] = lambda: None

    from src.api.schemas.user import UserResponse
    mocker.patch("src.api.schemas.user.UserResponse.from_orm", return_value=UserResponse(
        id=user.id, email=user.email, full_name=user.full_name, tier="enterprise",
        is_active=True, is_verified=True, is_mfa_enabled=False, created_at=datetime.now()
    ))
    
    response = client.get(f"/users/{user.id}")
    assert response.status_code == 200
    
    # Reset
    app.dependency_overrides[get_current_active_user] = get_mock_user

def test_list_users_enterprise(mocker):
    # Same for list_users
    for route in app.routes:
        if route.path == "/users":
            for dep in route.dependencies:
                app.dependency_overrides[dep.dependency] = lambda: None

    mock_db = mocker.Mock()
    app.dependency_overrides[get_db] = lambda: mock_db
    
    user = get_mock_user()
    mock_db.query.return_value.count.return_value = 1
    # Mock the chain: query().offset().limit().all()
    mock_db.query.return_value.filter.return_value = mock_db.query.return_value # search case
    mock_db.query.return_value.offset.return_value.limit.return_value.all.return_value = [user]
    
    from src.api.schemas.user import UserResponse
    mocker.patch("src.api.schemas.user.UserResponse.from_orm", return_value=UserResponse(
        id=user.id, email=user.email, full_name=user.full_name, tier="free",
        is_active=True, is_verified=True, is_mfa_enabled=False, created_at=datetime.now()
    ))
    
    response = client.get("/users?search=test")
    assert response.status_code == 200
    assert response.json()["pagination"]["total"] == 1
