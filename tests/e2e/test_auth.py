import os

import httpx
import pytest

# Base URL for the services
# In bridge network, we use service names. On host, we use localhost.
AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://auth-service:3001")
API_URL = os.getenv("API_URL", "http://api:8000")

@pytest.mark.asyncio
async def test_auth_service_health():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{AUTH_SERVICE_URL}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "operational"

@pytest.mark.asyncio
async def test_api_health():
    async with httpx.AsyncClient() as client:
        # In dev mode, API might be on 8000 or 8008. The gateway handles federation.
        response = await client.get(f"{API_URL}/health")
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_full_auth_flow_via_api():
    """
    Test user registration and login directly via the Auth service API.
    """
    unique_email = f"py_test_{os.urandom(4).hex()}@example.com"
    password = "SecurePassword123!"
    name = "Python Test User"

    async with httpx.AsyncClient() as client:
        # 1. Register
        reg_response = await client.post(
            f"{AUTH_SERVICE_URL}/api/auth/signup/email",
            json={
                "email": unique_email,
                "password": password,
                "name": name
            }
        )
        assert reg_response.status_code in [200, 201]

        # 2. Login
        login_response = await client.post(
            f"{AUTH_SERVICE_URL}/api/auth/sign-in/email",
            json={
                "email": unique_email,
                "password": password
            }
        )
        assert login_response.status_code == 200
        auth_data = login_response.json()
        assert "token" in auth_data or "session" in auth_data
        
        # Extracted token for subsequent API calls
        token = auth_data.get("token") or auth_data.get("session", {}).get("token")
        
        # 3. Verify access to a protected route (if any)
        # Assuming the API has a protected /me or similar
        # For now, we just verify we got a token.
        assert token is not None
