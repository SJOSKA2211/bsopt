from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.auth.auth import AuthService
from src.database.models import User


@pytest.fixture
def auth_service():
    return AuthService()


@pytest.mark.asyncio
async def test_argon2id_hashing_verification(auth_service):
    """Unit test for Argon2id hashing/verification."""
    password = "StrongPassword123!"
    hashed = auth_service.hash_password(password)

    # Argon2id hashes start with $argon2id$
    assert hashed.startswith("$argon2id$")
    assert auth_service.verify_password(password, hashed) is True
    assert auth_service.verify_password("wrong_password", hashed) is False


@pytest.mark.asyncio
async def test_jwt_signing_validation(auth_service):
    """Unit test for JWT signing/validation."""
    user_id = "user-123"
    email = "user@example.com"
    tier = "pro"

    token_pair = auth_service.create_token_pair(user_id, email, tier)

    assert token_pair.access_token is not None
    assert token_pair.refresh_token is not None

    # Validate access token
    token_data = auth_service.decode_token(token_pair.access_token)
    assert token_data.user_id == user_id
    assert token_data.email == email
    assert token_data.tier == tier
    assert token_data.token_type == "access"


@pytest.mark.asyncio
async def test_redis_token_revocation(auth_service):
    """Mock Redis test for token revocation."""
    user_id = "user-123"
    email = "user@example.com"
    tier = "pro"

    token_pair = auth_service.create_token_pair(user_id, email, tier)
    token_data = auth_service.decode_token(token_pair.access_token)
    jti = token_data.jti

    mock_redis = AsyncMock()
    # Mock exists to return 1 (True in Redis)
    mock_redis.exists.return_value = 1

    with patch("src.auth.auth.get_redis_client", return_value=mock_redis):
        is_revoked = await auth_service.is_token_revoked(jti)
        assert is_revoked is True
        mock_redis.exists.assert_called_once_with(f"blacklist:{jti}")


@pytest.mark.asyncio
async def test_authenticate_user_flow(auth_service):
    """Test the core authentication flow: authenticate_user -> verify_mfa -> create_token_pair."""
    password = "StrongPassword123!"
    hashed = auth_service.hash_password(password)

    user = User(
        id="user-123",
        email="user@example.com",
        hashed_password=hashed,
        tier="pro",
        mfa_enabled=False,
    )

    db = AsyncMock()
    # Mock the result of the query
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = user
    db.execute.return_value = mock_result

    # 1. Authenticate
    authenticated_user = await auth_service.authenticate_user(db, user.email, password)
    assert str(authenticated_user.id) == str(user.id)

    # 2. Verify MFA (disabled)
    mfa_ok = await auth_service.verify_mfa(authenticated_user, None)
    assert mfa_ok is True

    # 3. Create Token Pair
    token_pair = auth_service.create_token_pair(
        str(authenticated_user.id), authenticated_user.email, authenticated_user.tier
    )
    assert token_pair.access_token is not None
    assert token_pair.refresh_token is not None
