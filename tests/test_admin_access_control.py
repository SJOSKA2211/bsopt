
import pytest
from src.database.models import User
from src.security.auth import auth_service, TIER_ADMIN

@pytest.fixture
def enterprise_user(mock_db_session):
    user = User(
        email="enterprise@example.com",
        hashed_password="hashed_password",
        full_name="Enterprise User",
        tier="enterprise",
        is_active=True,
        is_verified=True
    )
    mock_db_session.add(user)
    return user

@pytest.fixture
def enterprise_token(enterprise_user):
    return auth_service.create_access_token(
        user_id=enterprise_user.id,
        email=enterprise_user.email,
        tier=enterprise_user.tier
    )

@pytest.fixture
def admin_user(mock_db_session):
    user = User(
        email="admin@example.com",
        hashed_password="hashed_password",
        full_name="Admin User",
        tier=TIER_ADMIN,
        is_active=True,
        is_verified=True
    )
    mock_db_session.add(user)
    return user

@pytest.fixture
def admin_token(admin_user):
    return auth_service.create_access_token(
        user_id=admin_user.id,
        email=admin_user.email,
        tier=admin_user.tier
    )

def test_enterprise_user_cannot_access_admin_endpoint(api_client, enterprise_token):
    """
    Verify that an 'enterprise' user CANNOT access admin-only endpoints.
    """
    headers = {"Authorization": f"Bearer {enterprise_token}"}

    # Try to list users (admin functionality)
    response = api_client.get("/api/v1/users", headers=headers)

    # Assert Forbidden
    assert response.status_code == 403

def test_admin_user_can_access_admin_endpoint(api_client, admin_token):
    """
    Verify that an 'admin' user CAN access admin-only endpoints.
    """
    headers = {"Authorization": f"Bearer {admin_token}"}

    # Try to list users (admin functionality)
    response = api_client.get("/api/v1/users", headers=headers)

    # Assert Success
    assert response.status_code == 200
    assert "items" in response.json()
