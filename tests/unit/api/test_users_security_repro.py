import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from api.index import app
from api.middleware.jwt_validator import require_auth
from src.auth.auth import get_current_active_user, get_current_user
from src.database import get_async_db
from src.database.models import User

client = TestClient(app)

@pytest.fixture(autouse=True)
def override_auth(request, free_user, admin_user):
    from src.auth.core.tokens import TokenData
    from datetime import UTC, datetime, timedelta

    current = admin_user if "admin" in request.node.name else free_user

    async def mocked_require_auth():
        return TokenData(
            user_id=str(current.id),
            email=current.email,
            tier=current.tier,
            token_type="access",
            exp=datetime.now(UTC) + timedelta(hours=1),
            iat=datetime.now(UTC),
            jti="test-jti",
            scopes=[]
        )

    app.dependency_overrides[get_current_active_user] = lambda: current
    app.dependency_overrides[get_current_user] = lambda: current
    app.dependency_overrides[require_auth] = mocked_require_auth
    yield
    app.dependency_overrides = {}


@pytest.fixture
def free_user():
    return User(
        id=uuid.uuid4(),
        email="free@example.com",
        full_name="Free User",
        tier="free",
        is_active=True,
        is_verified=True,
        mfa_enabled=False,
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
        mfa_enabled=False,
        created_at=datetime.now(UTC),
    )


def test_list_users_vulnerability_repro(free_user):
    """
    Verify vulnerability is fixed: Free user can NO LONGER list all users.
    Expected behavior (AFTER FIX): 403 Forbidden
    """
    app.dependency_overrides[get_current_active_user] = lambda: free_user
    app.dependency_overrides[get_current_user] = lambda: free_user
    mock_db = AsyncMock()
    app.dependency_overrides[get_async_db] = lambda: mock_db

    # Malicious attempt
    response = client.get("/api/v1/users")

    assert response.status_code == 403
    app.dependency_overrides = {}


def test_list_users_admin_access(admin_user, free_user):
    """
    Verify admin user can list users.
    """
    app.dependency_overrides[get_current_active_user] = lambda: admin_user
    app.dependency_overrides[get_current_user] = lambda: admin_user
    mock_db = AsyncMock()
    app.dependency_overrides[get_async_db] = lambda: mock_db

    mock_count_result = MagicMock()
    mock_count_result.scalar.return_value = 1
    mock_users_result = MagicMock()
    mock_users_result.scalars.return_value.all.return_value = [free_user]
    mock_db.execute.side_effect = [mock_count_result, mock_users_result]

    response = client.get("/api/v1/users")
    assert response.status_code == 200

    app.dependency_overrides = {}