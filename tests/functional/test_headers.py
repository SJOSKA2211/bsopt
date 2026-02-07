"""
HTTP Header and Cookie Functional Tests (Principles 64, 79, 95)
============================================================
"""

import pytest


@pytest.mark.asyncio
async def test_header_content_type_json(client):
    """79. Test Assertions: Check content types."""
    response = await client.get("/health")
    assert response.headers.get("content-type").startswith("application/json")

@pytest.mark.asyncio
async def test_header_security_and_server_info(client):
    """64. Test Assertions: Check header values."""
    response = await client.get("/health")
    # Verify standard security headers
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    # Ensure server info is not leaked or is generic
    server = response.headers.get("server", "").lower()
    assert "uvicorn" in server or server == "" # Standard/Generic allowed

@pytest.mark.asyncio
async def test_no_sensitive_cookies_set(client):
    """95. Test Assertions: Check cookie values."""
    response = await client.get("/health")
    # API shouldn't set cookies by default
    assert "set-cookie" not in response.headers
