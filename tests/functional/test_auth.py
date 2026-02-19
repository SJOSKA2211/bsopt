"""
Authentication Functional Tests (Refined Plan)
==============================================
"""

import pytest

from src.api.schemas.user import UserResponse


@pytest.mark.asyncio
async def test_register_success(client, mock_db, user_payload):
    """1. User Registration: Success case."""
    response = await client.post("/api/v1/auth/register", json=user_payload)
    assert response.status_code == 201
    assert response.json()["data"]["email"] == user_payload["email"]


@pytest.mark.asyncio
async def test_register_duplicate_email(client, user_payload):
    """2. User Registration: Duplicate email conflict."""
    await client.post("/api/v1/auth/register", json=user_payload)
    # Second attempt
    response = await client.post("/api/v1/auth/register", json=user_payload)
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_login_success(client, mock_db, user_payload):
    """3. User Login: Success case."""
    # Setup: Register and Verify
    reg_res = await client.post("/api/v1/auth/register", json=user_payload)
    user_id = reg_res.json()["data"]["user_id"]
    db_user = mock_db.query(None).filter(f"id == '{user_id}'").first()
    db_user.is_verified = True

    # Test
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": user_payload["email"], "password": user_payload["password"]},
    )
    assert response.status_code == 200
    assert "access_token" in response.json()["data"]


@pytest.mark.asyncio
async def test_login_invalid_credentials(client):
    """7. User Login: Invalid credentials."""
    response = await client.post(
        "/api/v1/auth/login", json={"email": "invalid@example.com", "password": "wrong"}
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_protected_endpoint_unauthorized(client):
    """7. Protected Endpoint: Unauthorized access."""
    response = await client.get("/api/v1/users/me")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_protected_endpoint_valid_token(client, mock_db, user_payload):
    """4. Protected Endpoint: Valid token."""
    # Setup
    reg_res = await client.post("/api/v1/auth/register", json=user_payload)
    user_id = reg_res.json()["data"]["user_id"]
    db_user = mock_db.query(None).filter(f"id == '{user_id}'").first()
    db_user.is_verified = True

    login_res = await client.post(
        "/api/v1/auth/login",
        json={"email": user_payload["email"], "password": user_payload["password"]},
    )
    token = login_res.json()["data"]["access_token"]

    # Test
    response = await client.get(
        "/api/v1/users/me", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    # 47. Schema validation
    UserResponse(**response.json()["data"])
