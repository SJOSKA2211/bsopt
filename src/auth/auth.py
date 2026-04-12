"""
Unified Authentication Service.

Consolidates login, MFA, and asymmetric JWT logic into a high-performance,
zero-trust compliant service.
"""

"""
Unified Authentication Service.

Consolidates login, MFA, and asymmetric JWT logic into a high-performance,
zero-trust compliant service.
"""

import hashlib
import logging
import secrets
from datetime import UTC, datetime

# Removed: from cachetools import TTLCache
from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from src.database import get_async_db
from src.database.models import APIKey, OAuth2Client, User
# Removed: from src.shared.utils.cache import get_redis_client # No longer directly used here

# Import the new centralized cache service
from src.common.caching import centralized_cache_service

# Removed: TokenData, TokenPair from tokens, will use token_service.TokenData and token_service.TokenPair
from src.auth.core.hashing import hasher
from src.auth.core.mfa import mfa_service
from src.auth.core.sessions import session_service
from src.auth.core.tokens import token_service
from src.auth.core.webauthn import webauthn_service

# Removed: High-performance local caches for FastAPI dependencies
# user_local_cache = TTLCache(maxsize=10000, ttl=60)  # 1 minute local TTL for users
# api_key_local_cache = TTLCache(maxsize=10000, ttl=60)  # 1 minute local TTL for API keys


class AuthService:
    """
    Unified Authentication Service.
    Delegates to modular core services for zero-trust compliance.
    """

    def __init__(self):
        # Maintain public API for backward compatibility while delegating
        self.hasher = hasher
        self.tokens = token_service
        self.mfa = mfa_service
        self.sessions = session_service

    # --- Password Logic (Delegated) ---

    def hash_password(self, password: str) -> str:
        return self.hasher.hash_password(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.hasher.verify_password(plain_password, hashed_password)

    def needs_rehash(self, hashed_password: str) -> bool:
        return self.hasher.needs_rehash(hashed_password)

    # --- Token Logic (Delegated) ---

    def create_token_pair(
        self, user_id: str, email: str, tier: str, scopes: list[str] = []
    ) -> token_service.TokenPair: # Use token_service.TokenPair for explicit typing
        return self.tokens.create_token_pair(user_id, email, tier, scopes)

    def decode_token(self, token: str) -> token_service.TokenData: # Use token_service.TokenData for explicit typing
        return self.tokens.decode_token(token)

    def generate_reset_token(self) -> str:
        return secrets.token_urlsafe(32)

    def generate_verification_token(self) -> str:
        return secrets.token_urlsafe(32)

    # --- MFA Logic (Delegated) ---

    def generate_mfa_secret(self) -> str:
        return self.mfa.generate_mfa_secret()

    def encrypt_mfa_secret(self, secret: str) -> str:
        return self.mfa.encrypt_mfa_secret(secret)

    def decrypt_mfa_secret(self, encrypted_secret: str) -> str:
        return self.mfa.decrypt_mfa_secret(encrypted_secret)

    def get_totp_uri(self, email: str, secret: str) -> str:
        return self.mfa.get_totp_uri(email, secret)

    def verify_mfa_code(self, secret: str, code: str) -> bool:
        return self.mfa.verify_mfa_code(secret, code)

    async def verify_mfa(self, user: User, code: str | None) -> bool:
        if not user.mfa_enabled:
            return True
        if not code:
            return False
        try:
            # Decrypt the MFA secret. This might fail if the stored secret is corrupted.
            secret = self.mfa.decrypt_mfa_secret(user.mfa_secret)
            # Verify the provided code against the decrypted secret.
            # This might fail if the code is invalid or the secret is malformed.
            return self.mfa.verify_mfa_code(secret, code)
        except ValueError as e: # Assuming ValueError for invalid code/secret format
            logger.warning(f"mfa_verification_error: Invalid MFA code or secret format. User: {user.id}. Error: {e}")
            return False
        except Exception as e: # Catch any other unexpected errors during decryption or verification
            logger.error(f"unexpected_mfa_verification_error: An unexpected error occurred during MFA verification. User: {user.id}. Error: {e}", exc_info=True)
            return False

    # --- Session & Revocation (Delegated) ---

    async def is_token_revoked(self, jti: str) -> bool:
        return await self.sessions.is_token_revoked(jti)

    async def revoke_token(self, token: str) -> None:
        await self.sessions.revoke_token(token)

    async def invalidate_token(self, token: str) -> None:
        """Alias for revoke_token for backward compatibility."""
        await self.revoke_token(token)

    async def validate_token(self, token: str) -> token_service.TokenData: # Use token_service.TokenData for explicit typing
        """
        High-performance token validation.
        """
        # 1. Fast Path (Redis - handled by centralized cache)
        # cached = await self.sessions.get_cached_session(token) # This is now handled by centralized_cache_service
        # if cached:
        #     return cached

        # Use the centralized cache service for token validation cache
        # Assuming centralized_cache_service has a method to get/set token data
        token_data = await centralized_cache_service.get_token_data_cached(token) 
        if token_data:
            return token_data

        # 2. Asymmetric JWT Validation
        token_data = self.decode_token(token)

        # 3. Revocation Check
        if token_data.jti and await self.sessions.is_token_revoked(token_data.jti):
            raise HTTPException(status_code=401, detail="Token revoked")

        # 4. Cache for future (handled by centralized cache service)
        # await self.sessions.cache_session(token, token_data)
        await centralized_cache_service.set_token_data_cached(token, token_data) 
        
        return token_data

    # --- Core Auth Flow ---

    async def authenticate_user(self, db: AsyncSession, email: str, password: str) -> User:
        """
        Authenticate a user with timing attack protection.
        """
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if not user or not user.hashed_password:
            # Timing attack protection
            await run_in_threadpool(self.hasher.verify_password, password, self.hasher.DUMMY_HASH)
            raise HTTPException(status_code=401, detail="Invalid credentials")

        password_matches = await run_in_threadpool(
            self.hasher.verify_password, password, user.hashed_password
        )
        if not password_matches:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        if self.hasher.needs_rehash(user.hashed_password):
            user.hashed_password = self.hasher.hash_password(password)

        return user

    # --- OAuth2 Client Logic ---

    async def authenticate_client(
        self, db: AsyncSession, client_id: str, client_secret: str
    ) -> OAuth2Client:
        """Authenticate a confidential OAuth2 client."""
        result = await db.execute(select(OAuth2Client).where(OAuth2Client.client_id == client_id))
        client = result.scalar_one_or_none()

        if not client or client.client_secret != client_secret:
            raise HTTPException(status_code=401, detail="Invalid client credentials")

        return client

    def create_client_credentials_token(self, client: OAuth2Client, scopes: list[str]) -> token_service.TokenPair: # Use token_service.TokenPair for explicit typing
        """Create a token for Client Credentials flow."""
        allowed_scopes = set(client.scopes or [])
        requested_scopes = set(scopes)
        if not requested_scopes.issubset(allowed_scopes):
            raise HTTPException(status_code=400, detail="Invalid scope requested")

        return self.tokens.create_client_credentials_token(client, scopes)

    # --- WebAuthn / Passkey (Delegated) ---

    def get_webauthn_registration_options(
        self, user_id: str, email: str, existing_credentials: list[bytes] = []
    ):
        return webauthn_service.get_registration_options(user_id, email, existing_credentials)

    def verify_webauthn_registration(self, registration_response: dict, expected_challenge: str):
        return webauthn_service.verify_registration(registration_response, expected_challenge)

    def get_webauthn_authentication_options(self, allow_credentials: list[bytes] = []):
        return webauthn_service.get_authentication_options(allow_credentials)

    def verify_webauthn_authentication(
        self,
        authentication_response: dict,
        expected_challenge: str,
        credential_public_key: bytes,
        credential_current_sign_count: int,
    ):
        return webauthn_service.verify_authentication(
            authentication_response,
            expected_challenge,
            credential_public_key,
            credential_current_sign_count,
        )

    # --- mTLS Support ---

    def verify_mtls(self, request: Request) -> bool:
        """
        Verify mTLS headers from trusted proxies.
        """
        client_verify = request.headers.get("X-SSL-Client-Verify")
        if client_verify != "SUCCESS":
            logger.warning("mtls_verification_failed", status=client_verify)
            return False

        return True

# Global instance
auth_service = AuthService()


# Removed: TokenBlacklistShim and token_blacklist global instance as it's legacy and uses revoke_token directly.
# The session_service.revoke_token is called directly by auth_service.revoke_token.


def get_auth_service() -> AuthService:
    return auth_service


# --- FastAPI Dependencies ---


async def get_token_from_header(
    credentials: HTTPAuthorizationCredentials | None = Depends(security_scheme),
) -> str | None:
    if credentials:
        return credentials.credentials
    return None


async def get_current_user(
    request: Request,
    token: str | None = Depends(get_token_from_header),
    db: AsyncSession = Depends(get_async_db),
    service: AuthService = Depends(get_auth_service),
) -> User:
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    token_data = await service.validate_token(token) # This will now use the centralized cache
    user_id = token_data.user_id

    # Use the centralized cache service for user retrieval
    cached_user_data = await centralized_cache_service.get_user_cached(user_id)
    if cached_user_data:
        user = User(**cached_user_data) # Assuming cached_user_data is a dict compatible with User model
        request.state.user = user
        return user

    # 3. Slow Path: DB
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    # Update caches using the centralized service
    user_dict_for_cache = {
        "id": str(user.id),
        "email": user.email,
        "tier": user.tier,
        "is_active": user.is_active,
        "is_verified": user.is_verified,
        "mfa_enabled": user.mfa_enabled,
    }
    await centralized_cache_service.set_user_cached(user_id, user_dict_for_cache)

    request.state.user = user
    return user


async def get_current_active_user(
    user: User = Depends(get_current_user),
) -> User:
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")
    return user


class RoleChecker:
    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = allowed_roles

    async def __call__(self, user: User = Depends(get_current_active_user)):
        user_roles = [user.tier]
        if user.tier == "enterprise":
            user_roles.append("admin")

        if not set(self.allowed_roles).intersection(user_roles):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user


async def get_api_key(
    request: Request,
    api_key: str | None = Depends(api_key_header),
    db: AsyncSession = Depends(get_async_db),
) -> User | None:
    if not api_key:
        return None

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    # Use the centralized cache service for API key retrieval
    cached_api_key_data = await centralized_cache_service.get_api_key_cached(key_hash)
    if cached_api_key_data:
        # Update last_used_at in distributed cache asynchronously
        await centralized_cache_service.update_api_key_last_used(key_hash)
        # Assuming cached_api_key_data contains user details to construct a User object
        return User(**cached_api_key_data)

    # 3. Slow path: DB
    from sqlalchemy.orm import joinedload

    result = await db.execute(
        select(APIKey)
        .options(joinedload(APIKey.user))
        .where(APIKey.key_hash == key_hash, APIKey.is_active)
    )
    key_record = result.scalar_one_or_none()

    if not key_record:
        return None

    user = key_record.user
    
    # Construct response data for caching
    user_dict_for_cache = {
        "id": str(user.id),
        "email": user.email,
        "tier": user.tier,
        "is_active": user.is_active,
        "is_verified": user.is_verified,
        "mfa_enabled": user.mfa_enabled,
        # You might want to include key-specific details if needed by the caller, e.g., key_name
        "key_name": key_record.name, 
    }

    # Update caches using the centralized service
    await centralized_cache_service.set_api_key_cached(key_hash, user_dict_for_cache)
    await centralized_cache_service.update_api_key_last_used(key_hash)

    return User(**user_dict_for_cache)


async def get_current_user_flexible(
    request: Request,
    token: str | None = Depends(get_token_from_header),
    api_key_user: User | None = Depends(get_api_key),
    db: AsyncSession = Depends(get_async_db),
    service: AuthService = Depends(get_auth_service),
) -> User | None:
    if api_key_user:
        return api_key_user

    if token:
        try:
            # This will use the centralized cache via service.validate_token
            return await get_current_user(request, token, db, service) 
        except HTTPException:
            pass

    return None
