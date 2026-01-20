import pytest
from unittest.mock import AsyncMock, patch
from jose import jwt
from fastapi import HTTPException
from src.auth.security import get_jwks, verify_token, RoleChecker, jwks_cache
import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import base64

# Helper to generate RSA keys for testing
def generate_rsa_keys():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()
    
    # Export to PEM
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    # Get n and e for JWKS
    numbers = public_key.public_numbers()
    n = base64.urlsafe_b64encode(numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8, 'big')).decode('utf-8').rstrip('=')
    e = base64.urlsafe_b64encode(numbers.e.to_bytes((numbers.e.bit_length() + 7) // 8, 'big')).decode('utf-8').rstrip('=')
    
    return private_pem, n, e

PRIVATE_PEM, N, E = generate_rsa_keys()

MOCK_KEY = {
    "kty": "RSA",
    "kid": "test_kid",
    "use": "sig",
    "n": N,
    "e": E
}
MOCK_JWKS = {"keys": [MOCK_KEY]}

@pytest.mark.asyncio
async def test_get_jwks_success():
    jwks_cache.clear()
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = AsyncMock(spec=httpx.Response)
        mock_get.return_value.json.return_value = MOCK_JWKS
        mock_get.return_value.status_code = 200
        
        keys = await get_jwks()
        assert keys == MOCK_JWKS
        assert "keys" in jwks_cache
        
        keys_cached = await get_jwks()
        assert keys_cached == MOCK_JWKS
        assert mock_get.call_count == 1

@pytest.mark.asyncio
async def test_verify_token_success():
    payload = {"sub": "user123", "aud": "account", "realm_access": {"roles": ["admin"]}}
    token = jwt.encode(payload, PRIVATE_PEM, algorithm="RS256", headers={"kid": "test_kid"})
    
    with patch("src.auth.security.get_jwks", return_value=MOCK_JWKS):
        verified_payload = await verify_token(token)
        assert verified_payload["sub"] == "user123"

@pytest.mark.asyncio
async def test_verify_token_invalid_kid():
    token = jwt.encode({"sub": "user123"}, "secret", algorithm="HS256", headers={"kid": "wrong_kid"})
    with patch("src.auth.security.get_jwks", return_value=MOCK_JWKS):
        with pytest.raises(HTTPException) as exc:
            await verify_token(token)
        assert exc.value.status_code == 401

@pytest.mark.asyncio
async def test_verify_token_jwt_error():
    # Token with invalid signature
    token = jwt.encode({"sub": "user123"}, "wrong_secret", algorithm="HS256", headers={"kid": "test_kid"})
    with patch("src.auth.security.get_jwks", return_value=MOCK_JWKS):
        with pytest.raises(HTTPException) as exc:
            await verify_token(token)
        assert exc.value.status_code == 401

def test_role_checker_success():
    checker = RoleChecker(allowed_roles=["admin"])
    payload = {"sub": "user123", "realm_access": {"roles": ["user", "admin"]}}
    result = checker(payload)
    assert result == payload

def test_role_checker_failure():
    checker = RoleChecker(allowed_roles=["admin"])
    payload = {"sub": "user123", "realm_access": {"roles": ["user"]}}
    with pytest.raises(HTTPException) as exc:
        checker(payload)
    assert exc.value.status_code == 403