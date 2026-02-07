import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from src.database.models import User


@pytest.fixture
def mock_db_with_users(mock_db_session):
    # Add some users to the mock DB
    user1 = User(
        id=uuid.uuid4(),
        email="u1@example.com",
        full_name="User One",
        is_active=True,
        tier="free",
        created_at=datetime.now(UTC),
        hashed_password="hash",
        mfa_secret=None,
    )
    user1.is_verified = True
    user1.is_mfa_enabled = False
    user1.updated_at = datetime.now(UTC)

    user2 = User(
        id=uuid.uuid4(),
        email="u2@example.com",
        full_name="User Two",
        is_active=True,
        tier="pro",
        created_at=datetime.now(UTC),
        hashed_password="hash",
        mfa_secret=None,
    )
    user2.is_verified = True
    user2.is_mfa_enabled = False
    user2.updated_at = datetime.now(UTC)

    mock_db_session.add(user1)
    mock_db_session.add(user2)
    return mock_db_session


def test_list_users_no_auth(api_client):
    response = api_client.get("/api/v1/users")
    assert response.status_code == 401


@patch("src.auth.providers.auth_registry.verify_any", new_callable=AsyncMock)
def test_list_users_non_admin(mock_verify, api_client, mock_db_with_users):
    # Mock a non-admin user
    mock_verify.return_value = {
        "sub": str(uuid.uuid4()),
        "email": "user@example.com",
        "roles": ["user"],
        "realm_access": {"roles": ["user"]},
    }

    headers = {"Authorization": "Bearer non-admin-token"}
    response = api_client.get("/api/v1/users", headers=headers)

    # NOW SECURE: Should return 403
    assert response.status_code == 403


@patch("src.auth.providers.auth_registry.verify_any", new_callable=AsyncMock)
def test_list_users_admin(mock_verify, api_client, mock_db_with_users):
    # Mock an admin user
    mock_verify.return_value = {
        "sub": str(uuid.uuid4()),
        "email": "admin@example.com",
        "roles": ["admin"],
        "realm_access": {"roles": ["admin"]},
    }

    headers = {"Authorization": "Bearer admin-token"}
    response = api_client.get("/api/v1/users", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    # MockQuery.all() returns all users in _users_by_email, so at least 2
    assert len(data["items"]) >= 2
