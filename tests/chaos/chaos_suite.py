"""
Chaos & Edge-Case Testing Suite (Phase 4).
Simulates adversarial inputs and network instability.
"""

import pytest
import httpx
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

@pytest.mark.asyncio
async def test_api_sqli_injection():
    """Simulate SQL injection attempts (Axiom: Chaos Phase 4)."""
    payloads = [
        "' OR 1=1 --",
        "\"; DROP TABLE users; --",
        "' UNION SELECT NULL, NULL, NULL --"
    ]
    
    async with httpx.AsyncClient() as client:
        for p in payloads:
            # Attempt injection on a search/filter endpoint
            response = await client.get(f"{API_URL}/market/historical-data", params={"symbol": p})
            # We expect a 400 Bad Request or 200 OK with empty result, but NEVER a 500 or executed query
            assert response.status_code in [200, 400, 422]
            if response.status_code == 200:
                data = response.json()
                assert len(data) == 0 or isinstance(data, dict)

@pytest.mark.asyncio
async def test_api_xss_injection():
    """Simulate Cross-Site Scripting attempts (Axiom: Chaos Phase 4)."""
    payload = "<script>alert('xss')</script>"
    
    async with httpx.AsyncClient() as client:
        # Attempt to create a portfolio with XSS payload
        # This requires auth, so we skip if no token
        token = os.getenv("TEST_TOKEN")
        if not token:
            pytest.skip("No TEST_TOKEN provided for XSS injection test")
            
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.post(
            f"{API_URL}/portfolios/", 
            json={"name": payload, "cash": 1000},
            headers=headers
        )
        
        # We expect the API to either reject it or sanitize it
        assert response.status_code in [201, 400, 422]
        if response.status_code == 201:
            data = response.json()
            # If accepted, ensure it didn't execute (though APIs usually just store)
            # The frontend should sanitize, but the API can too.
            assert "<script>" not in data["name"]

@pytest.mark.asyncio
async def test_api_malformed_json():
    """Simulate malformed JSON payloads (Axiom: Chaos Phase 4)."""
    async with httpx.AsyncClient() as client:
        headers = {"Content-Type": "application/json"}
        response = await client.post(
            f"{API_URL}/portfolios/", 
            content='{"name": "test", "cash": 1000,}', # Invalid trailing comma
            headers=headers
        )
        assert response.status_code == 400 or response.status_code == 422
