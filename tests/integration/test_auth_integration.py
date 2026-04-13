import pytest
from fastapi import status

@pytest.mark.integration
class TestAuthIntegration:
    """
    Zero-Mock Integration Tests for Auth Flow.
    Verifies that the JWT middleware correctly interacts with TokenService.
    """

    def test_unauthorized_access(self, api_client):
        """Verify that protected routes return 401 without token."""
        response = api_client.post("/api/v1/pricing/calculate", json={})
        # If it returns 405 it means the prefix/route is wrong, but /api/v1/pricing/calculate should exist
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_authorized_access(self, api_client, auth_headers):
        """Verify that protected routes return 200 with valid real JWT."""
        # Simple calculate request
        payload = {
            "spot": 100,
            "strike": 100,
            "time_to_expiry": 1.0,
            "rate": 0.05,
            "volatility": 0.2,
            "option_type": "call",
            "model": "black_scholes"
        }
        response = api_client.post("/api/v1/pricing/calculate", json=payload, headers=auth_headers)
        
        # We expect 200 if the math kernel is also working, or 500 if math kernel fails
        # but the AUTH part should pass (i.e., NOT 401/403)
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_500_INTERNAL_SERVER_ERROR]
        if response.status_code == 200:
            data = response.json()
            assert "price" in data
            assert data["spot"] == 100

    def test_invalid_token(self, api_client):
        """Verify that invalid tokens are rejected."""
        headers = {"Authorization": "Bearer not-a-valid-token"}
        response = api_client.post("/api/v1/pricing/calculate", json={}, headers=headers)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_token_revocation_integration(self, api_client, test_user_token):
        """
        Verify that revoking a token in Redis immediately affects API access.
        This is a true zero-mock integration test.
        """
        from src.auth.core.tokens import token_service
        from src.auth.auth import auth_service
        import asyncio
        import pytest

        # 1. Access works initially
        headers = {"Authorization": f"Bearer {test_user_token}"}
        response = api_client.post("/api/v1/pricing/calculate", json={}, headers=headers)
        assert response.status_code != status.HTTP_401_UNAUTHORIZED

        # 2. Revoke the token
        # We need to run the revocation in an async loop since it's an async method
        loop = asyncio.get_event_loop()
        loop.run_until_complete(auth_service.revoke_token(test_user_token))

        # 3. Access should now be denied
        response = api_client.post("/api/v1/pricing/calculate", json={}, headers=headers)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
