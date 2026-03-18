import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.database.models import User
from src.auth.auth import auth_service, token_blacklist


@pytest.mark.asyncio
async def test_token_creation_and_decoding():
    user_id = str(uuid.uuid4())
    email = "test@example.com"
    tier = "pro"

    pair = auth_service.create_token_pair(user_id, email, tier)
    assert pair.access_token is not None
    assert pair.refresh_token is not None

    # Decode access token
    data = auth_service.decode_token(pair.access_token)
    assert data.user_id == user_id
    assert data.email == email
    assert data.tier == tier
    assert data.token_type == "access"


@pytest.mark.asyncio
async def test_token_blacklist():
    jti = "test-jti"
    exp = datetime.now(UTC) + timedelta(hours=1)

    # Local set-based blacklist (no redis)
    await token_blacklist.initialize(redis_client=None)
    await token_blacklist.add(jti, exp)
    assert await token_blacklist.contains(jti) is True
    assert await token_blacklist.contains("other") is False


@pytest.mark.asyncio
async def test_authenticate_user_success():
    db = MagicMock()
    user = User(id=uuid.uuid4(), email="test@example.com", hashed_password="hashed")
    db.query.return_value.filter.return_value.first.return_value = user

    with patch("src.auth.auth.password_service.verify_password", return_value=True):
        with patch("src.auth.auth.password_service.needs_rehash", return_value=False):
            authenticated = await auth_service.authenticate_user(
                db, "test@example.com", "pass", MagicMock()
            )
            assert authenticated == user


@pytest.mark.asyncio
async def test_authenticate_user_fail_password():
    db = MagicMock()
    user = User(id=uuid.uuid4(), email="test@example.com", hashed_password="hashed")
    db.query.return_value.filter.return_value.first.return_value = user

    with patch("src.auth.auth.password_service.verify_password", return_value=False):
        authenticated = await auth_service.authenticate_user(
            db, "test@example.com", "wrong", MagicMock()
        )
        assert authenticated is None


@pytest.mark.asyncio
async def test_validate_token_revoked():
    token = auth_service._create_token({"sub": "123", "jti": "revoked"}, timedelta(minutes=5))

    with patch("src.auth.auth.token_blacklist.contains", return_value=True):
        with pytest.raises(HTTPException) as exc:
            await auth_service.validate_token(token)
        assert exc.value.status_code == 401
        assert "revoked" in exc.value.detail
