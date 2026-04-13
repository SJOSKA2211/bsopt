import asyncio
import httpx
import pytest
import structlog

logger = structlog.get_logger(__name__)

# Integration Smoke Test for BSOPT Overhaul
# Verifies that all services are online and correctly communicating.

AUTH_API_URL = "http://auth_api:3001"
PRICING_API_URL = "http://pricing_api:8000"
NGINX_URL = "http://nginx:80"

@pytest.mark.asyncio
async def test_auth_api_liveness():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{AUTH_API_URL}/health/liveness")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

@pytest.mark.asyncio
async def test_auth_api_readiness():
    async with httpx.AsyncClient() as client:
        # This might fail initially if DB/Redis/Vault aren't ready
        # But we use retries or DependsOn in docker-compose
        response = await client.get(f"{AUTH_API_URL}/health/readiness")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_pricing_api_health():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{PRICING_API_URL}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_nginx_routing_frontend():
    async with httpx.AsyncClient() as client:
        response = await client.get(NGINX_URL)
        assert response.status_code == 200
        # Check for some frontend-specific content
        assert "BSOPT" in response.text or "Vite" in response.text

@pytest.mark.asyncio
async def test_nginx_routing_auth():
    async with httpx.AsyncClient() as client:
        # Nginx routes /auth/ to auth_api:3001/
        response = await client.get(f"{NGINX_URL}/auth/health/liveness")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

@pytest.mark.asyncio
async def test_nginx_routing_api():
    async with httpx.AsyncClient() as client:
        # Nginx routes /api/ to pricing_api:8000/
        response = await client.get(f"{NGINX_URL}/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
