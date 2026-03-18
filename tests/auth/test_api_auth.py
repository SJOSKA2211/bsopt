from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app
from src.auth.security import verify_token

client = TestClient(app)


# Mocking the security dependencies to test the endpoint logic and RBAC integration
def test_admin_only_success():
    # Mock verify_token to return a valid admin payload
    # Mock RoleChecker to allow access
    payload = {"sub": "admin_user", "realm_access": {"roles": ["admin"]}}

    with patch("src.api.main.verify_token", return_value=payload):
        # We need to override the dependency in the app
        app.dependency_overrides[verify_token] = lambda: payload
        # RoleChecker is also a dependency, but it uses verify_token
        # Since we override verify_token, RoleChecker will get the mocked payload

        response = client.get("/admin-only")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome, Admin"}

        # Clean up overrides
        app.dependency_overrides = {}


def test_admin_only_forbidden():
    payload = {"sub": "regular_user", "realm_access": {"roles": ["user"]}}

    with patch("src.api.main.verify_token", return_value=payload):
        app.dependency_overrides[verify_token] = lambda: payload

        response = client.get("/admin-only")
        # RoleChecker should raise 403
        assert response.status_code == 403

        app.dependency_overrides = {}


def test_admin_only_unauthorized():
    # No token provided
    response = client.get("/admin-only")
    # verify_token (Depends(oauth2_scheme)) will raise 401 if no header
    assert response.status_code == 401
