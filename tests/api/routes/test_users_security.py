
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
from datetime import datetime

# Set environment variables required by Settings
import os
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["JWT_SECRET"] = "test-secret-key-1234567890"
os.environ["DATABASE_URL"] = "postgresql://user:pass@localhost:5432/db"
os.environ["TESTING"] = "true"

# Mock expensive/complex dependencies
mock_ray = MagicMock()
sys.modules["ray"] = mock_ray

mock_torch = MagicMock()
class MockTensor:
    pass
mock_torch.Tensor = MockTensor
sys.modules["torch"] = mock_torch

# Patch RayOrchestrator to avoid import side effects
with patch("src.utils.distributed.RayOrchestrator.init") as mock_init:
    from src.api.main import app
    from src.database import get_db
    from src.auth.providers import auth_registry
    from src.database.models import User

# Test client fixture
@pytest.fixture
def client():
    return TestClient(app)

# Mock DB fixture
@pytest.fixture
def mock_db_session():
    mock_db = MagicMock()

    # Mock query chain for list_users
    query_mock = MagicMock()
    mock_db.query.return_value = query_mock
    query_mock.scalar.return_value = 10

    offset_mock = MagicMock()
    query_mock.offset.return_value = offset_mock
    limit_mock = MagicMock()
    offset_mock.limit.return_value = limit_mock

    # Fully populated user objects
    user1 = User(
        id="00000000-0000-0000-0000-000000000001",
        email="user1@example.com",
        tier="free",
        is_active=True,
        created_at=datetime.now(),
        full_name="User One"
    )
    user1.is_verified = True
    user1.is_mfa_enabled = False

    user2 = User(
        id="00000000-0000-0000-0000-000000000002",
        email="user2@example.com",
        tier="free",
        is_active=True,
        created_at=datetime.now(),
        full_name="User Two"
    )
    user2.is_verified = True
    user2.is_mfa_enabled = False

    limit_mock.all.return_value = [user1, user2]

    return mock_db

@pytest.fixture
def override_deps(mock_db_session):
    app.dependency_overrides[get_db] = lambda: mock_db_session
    yield
    app.dependency_overrides.clear()

@pytest.mark.asyncio
async def test_list_users_denied_for_non_admin(client, override_deps):
    """
    Ensure non-admin users cannot access the user list.
    """
    # Mock auth registry to return a non-admin user
    async def mock_verify_non_admin(token):
        return {"sub": "user-123", "email": "user@example.com", "roles": ["free"]}

    with patch.object(auth_registry, "verify_any", side_effect=mock_verify_non_admin):
        response = client.get("/api/v1/users", headers={"Authorization": "Bearer non-admin-token"})

        assert response.status_code == 403
        assert response.json()["detail"] == "Insufficient Permissions"

@pytest.mark.asyncio
async def test_list_users_allowed_for_admin(client, override_deps):
    """
    Ensure admin users CAN access the user list.
    """
    # Mock auth registry to return an admin user
    async def mock_verify_admin(token):
        return {"sub": "admin-123", "email": "admin@example.com", "roles": ["admin"]}

    with patch.object(auth_registry, "verify_any", side_effect=mock_verify_admin):
        response = client.get("/api/v1/users", headers={"Authorization": "Bearer admin-token"})

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) == 2
        assert data["items"][0]["email"] == "user1@example.com"
