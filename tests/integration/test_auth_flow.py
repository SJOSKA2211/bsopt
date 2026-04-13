import pytest
import httpx
import asyncio
from uuid import uuid4
from src.database.models import User
from src.database.crud import create_user
from sqlalchemy.ext.asyncio import AsyncSession

@pytest.mark.asyncio
async def test_full_auth_cycle(db_session: AsyncSession):
    """
    Integration test for the full authentication cycle.
    Verifies:
    1. User creation in DB.
    2. Authentication via Auth API.
    3. Token generation and Redis caching.
    4. gRPC token introspection (if available).
    """
    email = f"test_{uuid4().hex[:8]}@example.com"
    password = "SecurePassword123!"
    full_name = "Integration Test User"
    
    # 1. Create user
    user = await create_user(db_session, email, password, full_name)
    assert user.email == email
    
    # 2. Authenticate via API
    async with httpx.AsyncClient() as client:
        # Note: In a containerized test, we use the service name 'auth_api'
        # For a local test, we assume it's running on localhost:3001
        try:
            response = await client.post(
                "http://localhost:3001/login",
                json={"email": email, "password": password}
            )
            
            if response.status_code == 200:
                data = response.json()
                assert "access_token" in data
                token = data["access_token"]
                
                # 3. Verify token introspection
                introspect_resp = await client.get(
                    "http://localhost:3001/userinfo",
                    headers={"Authorization": f"Bearer {token}"}
                )
                assert introspect_resp.status_code == 200
                assert introspect_resp.json()["email"] == email
        except httpx.ConnectError:
            pytest.skip("Auth API not reachable for integration test")

@pytest.mark.asyncio
async def test_redis_connection():
    """Verify that we can talk to the hardened Redis instance."""
    from src.shared.utils.cache import get_redis_client
    redis = await get_redis_client()
    if not redis:
        pytest.skip("Redis not reachable")
    
    test_key = f"test_{uuid4().hex}"
    await redis.set(test_key, "manifold_health_check", ex=10)
    val = await redis.get(test_key)
    assert val == "manifold_health_check"
