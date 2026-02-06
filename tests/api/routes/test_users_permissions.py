import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from src.api.main import app
from src.database import get_db


# Mock DB dependency
def override_get_db():
    mock_db = MagicMock()
    mock_db.query.return_value.count.return_value = 1
    mock_db.query.return_value.scalar.return_value = 1
    mock_db.query.return_value.offset.return_value.limit.return_value.all.return_value = (
        []
    )
    return mock_db


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


def test_list_users_permissions():
    # Mock payload for a regular user
    regular_user_payload = {
        "sub": "user-123",
        "email": "user@example.com",
        "roles": ["free"],  # Not admin
        "realm_access": {"roles": ["free"]},
    }

    # Mock payload for an admin user
    admin_user_payload = {
        "sub": "admin-123",
        "email": "admin@example.com",
        "roles": ["admin"],  # Admin role
        "realm_access": {"roles": ["admin"]},
    }

    with patch(
        "src.auth.providers.auth_registry.verify_any", new_callable=AsyncMock
    ) as mock_verify:
        # Test 1: Regular User accessing list_users -> Should return 403 Forbidden
        mock_verify.return_value = regular_user_payload
        headers = {"Authorization": "Bearer regular-token"}
        response = client.get("/api/v1/users", headers=headers)
        assert response.status_code == 403

        # Test 2: Admin User accessing list_users -> Should return 200 OK
        mock_verify.return_value = admin_user_payload
        headers = {"Authorization": "Bearer admin-token"}
        response = client.get("/api/v1/users", headers=headers)
        assert response.status_code == 200
