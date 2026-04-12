from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.index import app
from src.auth.auth import get_current_active_user
from src.database.models import User

client = TestClient(app)


def test_admin_only_success():
    # Mock user with admin role (enterprise tier in this system)
    mock_user = MagicMock(spec=User)
    mock_user.id = "admin-123"
    mock_user.email = "admin@example.com"
    mock_user.tier = "enterprise"
    mock_user.is_active = True

    app.dependency_overrides[get_current_active_user] = lambda: mock_user

    try:
        response = client.get("/admin-only")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome, Admin"}
    finally:
        app.dependency_overrides = {}


def test_admin_only_forbidden():
    # Mock user with regular role
    mock_user = MagicMock(spec=User)
    mock_user.id = "user-123"
    mock_user.email = "user@example.com"
    mock_user.tier = "free"
    mock_user.is_active = True

    app.dependency_overrides[get_current_active_user] = lambda: mock_user

    try:
        response = client.get("/admin-only")
        # RoleChecker should raise 403
        assert response.status_code == 403
    finally:
        app.dependency_overrides = {}


def test_admin_only_unauthorized():
    # If we don't override, it will use real dependency which will fail with 401
    # as there is no token/header.
    response = client.get("/admin-only")
    assert response.status_code == 401
