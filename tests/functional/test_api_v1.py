import pytest
from httpx import AsyncClient, ASGITransport
from src.api.main import app
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 🚀 SINGULARITY: Clean health path
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "degraded"]

@pytest.mark.asyncio
async def test_pricing_rate_limit():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "spot": 100.0, "strike": 100.0, "time_to_expiry": 1.0,
            "volatility": 0.2, "rate": 0.05, "dividend_yield": 0.0,
            "option_type": "call", "model": "black_scholes"
        }

        # Mock cache to return None so it proceeds to pricing logic
        with patch("src.api.routes.pricing.pricing_cache.get_option_price", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None
            # 🚀 SINGULARITY: Standardized pricing path
            for _ in range(5):
                response = await ac.post("/pricing/price", json=payload)
                # We expect success or rate limit
                assert response.status_code in [200, 429]

@pytest.mark.asyncio
async def test_error_response_format():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 🚀 SINGULARITY: Standardized pricing path
        response = await ac.post("/pricing/price", json={})
    
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert data["error"] == "ValidationError"
    assert "message" in data
    assert "timestamp" in data
