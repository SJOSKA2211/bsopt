"""
Pricing Engine Functional Tests (Refined Plan)
==============================================
"""

import pytest


@pytest.mark.asyncio
async def test_pricing_valid_request(client):
    """5. Pricing Endpoint: Valid payload."""
    payload = {
        "spot": 100.0,
        "strike": 100.0,
        "time_to_expiry": 1.0,
        "volatility": 0.2,
        "rate": 0.05,
        "dividend_yield": 0.0,
        "option_type": "call",
        "model": "black_scholes",
    }
    response = await client.post("/api/v1/pricing/price", json=payload)
    assert response.status_code == 200
    assert "price" in response.json()["data"]


@pytest.mark.asyncio
async def test_pricing_invalid_payload(client):
    """7. Pricing Endpoint: Invalid payload (Schema violation)."""
    response = await client.post("/api/v1/pricing/price", json={})
    assert response.status_code == 422
