"""
Unified Authentication Service.

Consolidates login, MFA, and asymmetric JWT logic into a high-performance,
zero-trust compliant service.
"""

import hashlib
import logging
import secrets
from datetime import UTC, datetime

from cachetools import TTLCache
from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from src.database import get_async_db
from src.database.models import APIKey, OAuth2Client, User
from src.shared.utils.cache import get_redis_client

logger = logging.getLogger(__name__)

# Security schemes for FastAPI docs
security_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

from src.auth.core.hashing import hasher
from src.auth.core.mfa import mfa_service
from src.auth.core.sessions import session_service
from src.auth.core.tokens import TokenData, TokenPair, token_service
from src.auth.core.webauthn import webauthn_service

# High-performance local caches for FastAPI dependencies
user_local_cache = TTLCache(maxsize=10000, ttl=60)  # 1 minute local TTL for users
api_key_local_cache = TTLCache(maxsize=10000, ttl=60)  # 1 minute local TTL for API keys


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
    ) -> TokenPair:
        return self.tokens.create_token_pair(user_id, email, tier, scopes)

    def decode_token(self, token: str) -> TokenData:
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
            secret = self.mfa.decrypt_mfa_secret(user.mfa_secret)
            return self.mfa.verify_mfa_code(secret, code)
        except Exception as e:
            logger.error(f"mfa_verification_error: {e}")
            return False

    # --- Session & Revocation (Delegated) ---

    async def is_token_revoked(self, jti: str) -> bool:
        return await self.sessions.is_token_revoked(jti)

    async def revoke_token(self, token: str) -> None:
        await self.sessions.revoke_token(token)

    async def validate_token(self, token: str) -> TokenData:
        """
        High-performance token validation.
        """
        # 1. Fast Path (Redis)
        cached = await self.sessions.get_cached_session(token)
        if cached:
            return cached

        # 2. Asymmetric JWT Validation
        token_data = self.decode_token(token)

        # 3. Revocation Check
        if token_data.jti and await self.sessions.is_token_revoked(token_data.jti):
            raise HTTPException(status_code=401, detail="Token revoked")

        # 4. Cache for future
        await self.sessions.cache_session(token, token_data)
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

    def create_client_credentials_token(self, client: OAuth2Client, scopes: list[str]) -> TokenPair:
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


class TokenBlacklistShim:
    """Legacy shim for token blacklist operations."""

    async def initialize(self, redis_client=None):
        pass

    async def add(self, jti: str, exp: datetime):
        from src.auth.core.tokens import TokenData

        await session_service.revoke_token(
            TokenData(
                jti=jti,
                exp=exp,
                user_id="",
                email="",
                tier="",
                token_type="access",
                iat=datetime.now(),
            )
        )

    async def contains(self, jti: str) -> bool:
        return await session_service.is_token_revoked(jti)


TokenBlacklist = TokenBlacklistShim
token_blacklist = TokenBlacklistShim()

auth_service.token_blacklist = token_blacklist


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

    token_data = await service.validate_token(token)
    user_id = token_data.user_id

    # 1. Fast Path: Local Cache
    if user_id in user_local_cache:
        user_dict = user_local_cache[user_id]
        user = User(**user_dict)
        request.state.user = user
        return user

    # 2. Medium Path: Redis Cache
    from src.shared.utils.cache import db_cache

    try:
        cached_user = await db_cache.get_user(user_id)
        if cached_user:
            user_local_cache[user_id] = cached_user
            user = User(**cached_user)
            request.state.user = user
            return user
    except Exception:
        pass

    # 3. Slow Path: DB
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    # Update caches
    user_dict = {
        "id": str(user.id),
        "email": user.email,
        "tier": user.tier,
        "is_active": user.is_active,
        "is_verified": user.is_verified,
        "mfa_enabled": user.mfa_enabled,
    }
    user_local_cache[user_id] = user_dict
    await db_cache.set_user(user_id, user_dict)

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

    # 1. Fast path: Local Cache
    if key_hash in api_key_local_cache:
        user_dict = api_key_local_cache[key_hash]
        return User(**user_dict)

    from src.shared.utils.cache import db_cache

    # 2. Medium path: Redis Cache
    cached_data = await db_cache.get_api_key(key_hash)
    if cached_data:
        user_id = cached_data.get("user_id")
        cached_user = await db_cache.get_user(user_id)
        if cached_user:
            api_key_local_cache[key_hash] = cached_user
            # Buffer the last_used_at update in Redis
            redis = await get_redis_client()
            await redis.hset("api_key_last_used", key_hash, datetime.now(UTC).isoformat())
            return User(**cached_user)

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
    user_dict = {
        "id": str(user.id),
        "email": user.email,
        "tier": user.tier,
        "is_active": user.is_active,
        "is_verified": user.is_verified,
        "mfa_enabled": user.mfa_enabled,
    }

    # Update caches
    api_key_local_cache[key_hash] = user_dict
    await db_cache.set_api_key(
        key_hash,
        {
            "user_id": str(user.id),
            "email": user.email,
            "tier": user.tier,
            "key_name": key_record.name,
        },
    )

    # Async update for last_used_at
    key_record.last_used_at = datetime.now(UTC)
    await db.commit()

    return User(**user_dict)


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
            return await get_current_user(request, token, db, service)
        except HTTPException:
            pass

    return None
