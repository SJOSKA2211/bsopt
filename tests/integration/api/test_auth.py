import pytest
from httpx import AsyncClient

# Assuming these tests will run against the API service which calls Auth gRPC

# Marker for auth-related tests
pytestmark = pytest.mark.integration

async def test_auth_verify_valid_token(api_client: AsyncClient, test_user_token: str):
    """
    Tests the /api/v1/auth/verify endpoint with a valid token.
    """
    response = await api_client.get(
        "/api/v1/auth/verify",
        params={"token": test_user_token}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["user_id"] == "test-integration-user"
    assert data["email"] == "test@manifold.test"
    assert data["tier"] == "admin"
    assert "roles" in data
    assert "admin" in data["roles"]

async def test_auth_verify_invalid_token(api_client: AsyncClient):
    """
    Tests the /api/v1/auth/verify endpoint with an invalid token.
    """
    response = await api_client.get(
        "/api/v1/auth/verify",
        params={"token": "invalid-token-string"}
    )
    
    assert response.status_code == 401 # Unauthorized
    data = response.json()
    assert "Token is invalid or expired" in data["detail"]

async def test_auth_verify_missing_token(api_client: AsyncClient):
    """
    Tests the /api/v1/auth/verify endpoint when the token is missing.
    """
    response = await api_client.get("/api/v1/auth/verify")
    
    assert response.status_code == 400 # Bad Request
    data = response.json()
    assert "Token is required" in data["detail"]

# Note: Testing token expiration would require generating a token with a past expiry
# and possibly mocking time.datetime.now() or using a fixture to control expiry.
# For now, we focus on basic validation.

# Add tests for CreateTokenPair if exposed via API, or test gRPC directly if needed.
# As of now, CreateTokenPair is only in the Auth gRPC service and not exposed via FastAPI.
# If it were exposed, we'd test it similarly.
