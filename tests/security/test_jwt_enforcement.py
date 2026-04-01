import sys
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

# Mock msgspec before it's imported
sys.modules["msgspec"] = MagicMock()

# Mock the auth_service import inside the middleware
mock_auth = MagicMock()
mock_auth.auth_service.validate_token.return_value.user_id = "user123"
mock_auth.auth_service.validate_token.return_value.email = "test@example.com"
mock_auth.auth_service.validate_token.return_value.tier = "free"
sys.modules["src.auth.auth"] = mock_auth

# Mock config settings
mock_config = MagicMock()
mock_config.settings.ENVIRONMENT = "dev"
sys.modules["src.shared.config"] = mock_config

from api.middleware.security import JWTAuthenticationMiddleware  # noqa: E402

app = FastAPI()
app.add_middleware(JWTAuthenticationMiddleware)

@app.get("/protected")
def protected_endpoint():
    return {"message": "protected"}

@app.get("/api/v1/auth/verify-email")
def public_endpoint():
    return {"message": "public"}

client = TestClient(app)

def test_auth_enforced_in_dev():
    # Ensure environment is dev
    mock_config.settings.ENVIRONMENT = "dev"

    # Request without token to protected endpoint
    response = client.get("/protected")

    # Should be BLOCKED (401) now
    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication token missing"}

def test_public_path_allowed():
    # Request without token to public endpoint
    response = client.get("/api/v1/auth/verify-email")

    # Should be ALLOWED (200)
    assert response.status_code == 200
    assert response.json() == {"message": "public"}
