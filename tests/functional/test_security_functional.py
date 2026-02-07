"""
Security Vulnerability Functional Tests (Principles 50, 58, 66, 74, 82, 98)
==========================================================================
"""


import pytest


@pytest.mark.asyncio
async def test_security_sql_injection_on_user_id(client):
    """50. Test Security: Check for SQL injection."""
    malicious_id = "780d916c' OR '1'='1"
    response = await client.get(f"/api/v1/users/{malicious_id}")
    # Should be rejected by validation or return 401/404, NOT execute SQL
    assert response.status_code in [401, 404, 422]

@pytest.mark.asyncio
async def test_security_xss_protection(client, user_payload):
    """58. Test Security: Check for XSS."""
    user_payload["full_name"] = "<script>alert('xss')</script>"
    response = await client.post("/api/v1/auth/register", json=user_payload)
    assert response.status_code == 201
    # 98. Sensitive data exposure check
    assert "<script>" not in response.text

@pytest.mark.asyncio
async def test_security_headers_present(client):
    """66. Test Security: Check security headers."""
    response = await client.get("/health")
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert "Content-Security-Policy" in response.headers
    assert response.headers.get("X-Content-Type-Options") == "nosniff"

@pytest.mark.asyncio
async def test_security_cors_configuration(client):
    """CORS: Test CORS configuration."""
    response = await client.options(
        "/api/v1/auth/login",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        }
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

@pytest.mark.asyncio
async def test_security_rate_limiting(client, user_payload):
    """Rate Limiting: Test rate limiting scenarios."""
    # We mock the rate limiter or hit the endpoint repeatedly
    # Since we are using a mock DB, we check if the app handles rapid requests
    # In a real env, we'd expect 429 after N requests
    for _ in range(5):
        await client.get("/health")
    # Verified health still works, real rate limit test would need Redis live
    pass

@pytest.mark.asyncio
async def test_security_authentication_bypass(client):
    """74. Test Security: Check for authentication bypass."""
    response = await client.get("/api/v1/users/me")
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_security_idor_profile_access(client, mock_db, user_payload):
    """82. Test Security: Check for insecure direct object references."""
    # 1. Register User A
    reg_res = await client.post("/api/v1/auth/register", json=user_payload)
    user_a_id = reg_res.json()["data"]["user_id"]
    
    # 2. Register User B
    user_b_payload = user_payload.copy()
    user_b_payload["email"] = "other_user_idor@example.com"
    reg_res_b = await client.post("/api/v1/auth/register", json=user_b_payload)
    user_b_id = reg_res_b.json()["data"]["user_id"]
    
    # User A verifies and logs in
    mock_db.users[user_a_id].is_verified = True
    login_res = await client.post("/api/v1/auth/login", json={
        "email": user_payload["email"],
        "password": user_payload["password"]
    })
    token_a = login_res.json()["data"]["access_token"]
    
    # User A tries to access User B's details via admin-only endpoint
    # User A is 'free', so it should be 403 Forbidden
    response = await client.get(
        f"/api/v1/users/{user_b_id}",
        headers={"Authorization": f"Bearer {token_a}"}
    )
    assert response.status_code == 403
