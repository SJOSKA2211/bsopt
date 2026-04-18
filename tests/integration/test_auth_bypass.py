import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import User
from src.shared.config import settings


@pytest.mark.asyncio
async def test_auth_registration_email_bypass(db: AsyncSession, client: AsyncClient):
    """Verify that @Manifold.test users are automatically verified when the bypass is enabled.
    """
    # 1. Enable bypass temporarily
    settings.ALLOW_E2E_EMAIL_BYPASS = True

    unique_email = f"test-bypass-{int(pytest.approx(0, abs=10000))}@Manifold.test"
    payload = {"email": unique_email, "password": "SecurePass123!", "full_name": "Test Bypass"}

    # Register user
    response = await client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 201

    # 2. Verify user is verified in DB (Checking DB path directly via the session)
    # The register route might have committed, but we need to check if we can see it
    result = await db.execute(select(User).where(User.email == unique_email))
    user = result.scalar_one_or_none()

    if user:
        assert user.is_verified is True
        assert user.email.endswith("@Manifold.test")
    else:
        # Fallback: check via /auth/me if we can login
        login_res = await client.post(
            "/api/v1/auth/login", json={"email": unique_email, "password": payload["password"]},
        )
        assert login_res.status_code == 200
        token = login_res.json()["data"]["access_token"]

        me_res = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_res.status_code == 200
        assert me_res.json()["data"]["is_verified"] is True
