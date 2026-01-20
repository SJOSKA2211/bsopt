import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import HTTPException
from src.auth.security import get_jwks, verify_token, RoleChecker, jwks_cache
from jose import jwt, JWTError

@pytest.mark.asyncio
async def test_get_jwks_cache_miss():
    jwks_cache.clear()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"keys": [{"kid": "1"}]}
    
    with patch('httpx.AsyncClient.get', new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        keys = await get_jwks()
        assert keys == {"keys": [{"kid": "1"}]}
        assert "keys" in jwks_cache

@pytest.mark.asyncio
async def test_get_jwks_cache_hit():
    jwks_cache["keys"] = {"keys": [{"kid": "cached"}]}
    keys = await get_jwks()
    assert keys == {"keys": [{"kid": "cached"}]}

@pytest.mark.asyncio
async def test_verify_token_success():
    mock_jwks = {"keys": [{"kid": "k1", "kty": "RSA", "use": "sig", "n": "...", "e": "..."}]}
    token = "fake.token.payload"
    
    with patch('src.auth.security.get_jwks', return_value=mock_jwks):
        with patch('jose.jwt.get_unverified_header', return_value={"kid": "k1"}):
            with patch('jose.jwt.decode', return_value={"sub": "u1", "roles": ["admin"]}):
                payload = await verify_token(token)
                assert payload["sub"] == "u1"

@pytest.mark.asyncio
async def test_verify_token_no_rsa_key():
    mock_jwks = {"keys": [{"kid": "wrong", "kty": "RSA", "use": "sig", "n": "...", "e": "..."}]}
    token = "fake.token.payload"
    
    with patch('src.auth.security.get_jwks', return_value=mock_jwks):
        with patch('jose.jwt.get_unverified_header', return_value={"kid": "k1"}):
            with pytest.raises(HTTPException) as exc:
                await verify_token(token)
            assert exc.value.status_code == 401

@pytest.mark.asyncio
async def test_verify_token_decode_error():
    mock_jwks = {"keys": [{"kid": "k1", "kty": "RSA", "use": "sig", "n": "...", "e": "..."}]}
    token = "fake.token.payload"
    
    with patch('src.auth.security.get_jwks', return_value=mock_jwks):
        with patch('jose.jwt.get_unverified_header', return_value={"kid": "k1"}):
            with patch('jose.jwt.decode', side_effect=JWTError("Invalid token")):
                with pytest.raises(HTTPException) as exc:
                    await verify_token(token)
                assert exc.value.status_code == 401

def test_role_checker_success():
    checker = RoleChecker(allowed_roles=["admin"])
    token_payload = {"realm_access": {"roles": ["admin", "user"]}}
    assert checker(token_payload) == token_payload

def test_role_checker_denial():
    checker = RoleChecker(allowed_roles=["admin"])
    token_payload = {"realm_access": {"roles": ["user"]}}
    with pytest.raises(HTTPException) as exc:
        checker(token_payload)
    assert exc.value.status_code == 403
