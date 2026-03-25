import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.auth.auth import get_current_active_user
from src.database import get_db
from src.database.models import User

client = TestClient(app)

@pytest.fixture
def free_user():
    return User(
        id=uuid.uuid4(),
        email="free@example.com",
        full_name="Free User",
        tier="free",
        is_active=True,
        is_verified=True,
        is_mfa_enabled=False,
        created_at=datetime.now(UTC),
    )

@pytest.fixture
def admin_user():
    return User(
        id=uuid.uuid4(),
        email="admin@example.com",
        full_name="Admin User",
        tier="admin",
        is_active=True,
        is_verified=True,
        is_mfa_enabled=False,
        created_at=datetime.now(UTC),
    )

def test_list_users_vulnerability_repro(free_user):
    """
    Verify vulnerability is fixed: Free user can NO LONGER list all users.
    Expected behavior (AFTER FIX): 403 Forbidden
    """
    app.dependency_overrides[get_current_active_user] = lambda: free_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db

    # Mock query logic
    mock_query = mock_db.query.return_value
    mock_query.scalar.return_value = 1
    mock_query.offset.return_value.limit.return_value.all.return_value = [free_user]

    response = client.get("/api/v1/users")

    assert response.status_code == 403

    app.dependency_overrides = {}

def test_list_users_admin_access(admin_user, free_user):
    """
    Verify admin user can list users.
    """
    app.dependency_overrides[get_current_active_user] = lambda: admin_user
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db

    mock_query = mock_db.query.return_value
    mock_query.scalar.return_value = 1
    mock_query.offset.return_value.limit.return_value.all.return_value = [free_user]

    response = client.get("/api/v1/users")
    assert response.status_code == 200

    app.dependency_overrides = {}
