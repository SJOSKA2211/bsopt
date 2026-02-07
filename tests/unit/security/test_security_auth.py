from datetime import UTC, datetime

import pytest

from src.security.auth import AuthService, TokenBlacklist


@pytest.fixture
def auth_service():
    return AuthService()

def test_token_creation_and_decoding(auth_service):
    user_id = "test-user-id"
    email = "test@example.com"
    tier = "free"
    
    token = auth_service.create_access_token(user_id, email, tier)
    assert isinstance(token, str)
    
    data = auth_service.decode_token(token)
    assert data.user_id == user_id
    assert data.email == email
    assert data.tier == tier
    assert data.token_type == "access"

def test_refresh_token(auth_service):
    user_id = "test-user-id"
    email = "test@example.com"
    
    token = auth_service.create_refresh_token(user_id, email)
    data = auth_service.decode_token(token)
    assert data.token_type == "refresh"

@pytest.mark.asyncio
async def test_token_blacklist():
    blacklist = TokenBlacklist()
    jti = "test-jti"
    exp = datetime.now(UTC)
    
    await blacklist.add(jti, exp)
    assert await blacklist.contains(jti) is True
    assert not await blacklist.contains("other-jti")
    
    await blacklist.cleanup()
    assert await blacklist.contains(jti) is False

def test_decode_invalid_token(auth_service):
    with pytest.raises(pytest.importorskip("fastapi").HTTPException) as exc:
        auth_service.decode_token("invalid-token")
    assert exc.value.status_code == 401
