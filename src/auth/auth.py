"""
Unified Authentication Service.

Consolidates login, MFA, and asymmetric JWT logic into a high-performance,
zero-trust compliant service.
"""

import hashlib
import logging
import secrets
from datetime import UTC, datetime

import msgspec
import structlog
from src.shared.config import settings
from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from src.auth.core.hashing import hasher
from src.auth.core.mfa import mfa_service
from src.auth.core.sessions import session_service
from src.auth.core.tokens import TokenData, TokenPair, token_service
from src.auth.core.webauthn import webauthn_service
from src.auth.exceptions import (
    InsufficientPermissionsError,
    InvalidCredentialsError,
    TokenRevokedError,
)
from src.common.caching import centralized_cache_service
from src.database import get_async_db
from src.database.models import APIKey, OAuth2Client, User

logger = structlog.get_logger(__name__)

security_scheme = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class AuthService:
    """
    Unified Authentication Service.
    Delegates to modular core services for zero-trust compliance.
    """

    def __init__(self):
        self.hasher = hasher
        self.tokens = token_service
        self.mfa = mfa_service
        self.sessions = session_service

    @property
    def token_blacklist(self) -> session_service:
        return self.sessions

    def hash_password(self, password: str) -> str:
        return self.hasher.hash_password(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.hasher.verify_password(plain_password, hashed_password)

    def needs_rehash(self, hashed_password: str) -> bool:
        return self.hasher.needs_rehash(hashed_password)

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
            secret = self.decrypt_mfa_secret(user.mfa_secret)
            return self.verify_mfa_code(secret, code)
        except ValueError as e:
            logger.warning("mfa_verification_error", user_id=user.id, error=str(e))
            return False
        except Exception as e:
            logger.error("unexpected_mfa_verification_error", user_id=user.id, error=str(e), exc_info=True)
            return False

    async def is_token_revoked(self, jti: str) -> bool:
        return await self.sessions.is_token_revoked(jti)

    async def revoke_token(self, token: str) -> None:
        await self.sessions.revoke_token(token)

    async def invalidate_token(self, token: str) -> None:
        await self.revoke_token(token)

    async def validate_token(self, token: str) -> TokenData:
        """
        High-performance token validation.
        Uses gRPC for centralized validation if configured (Remote), 
        otherwise falls back to local asymmetric decoding (Lateral).
        """
        # 1. Check Distributed Cache First
        token_data = await centralized_cache_service.get_token_data_cached(token)
        if token_data:
            if isinstance(token_data, dict):
                token_data = msgspec.json.decode(msgspec.json.encode(token_data), type=TokenData)
            return token_data

        # 2. Centralized Validation (gRPC) if not the Auth Service itself
        if settings.AUTH_SERVICE_GRPC_URL and not os.getenv("IS_AUTH_SERVICE") == "true":
            from src.auth.grpc_client import auth_grpc_client
            
            grpc_resp = await auth_grpc_client.validate_token(token)
            if grpc_resp and grpc_resp.valid:
                token_data = TokenData(
                    user_id=grpc_resp.user_id,
                    email=grpc_resp.email,
                    tier=grpc_resp.tier,
                    token_type=grpc_resp.token_type,
                    exp=datetime.fromtimestamp(grpc_resp.expires_at, tz=UTC),
                    iat=datetime.fromtimestamp(grpc_resp.issued_at, tz=UTC),
                    jti=None,
                    scopes=list(grpc_resp.roles)
                )
                await centralized_cache_service.set_token_data_cached(token, token_data)
                return token_data
            elif grpc_resp and not grpc_resp.valid:
                raise TokenRevokedError()
            else:
                logger.warning("auth_grpc_fallback_to_local", token_truncated=token[:10])

        # 3. Lateral Validation (Local)
        token_data = self.decode_token(token)
        
        # 4. Consistency Check (Revocation)
        if token_data.jti and await self.sessions.is_token_revoked(token_data.jti):
            raise TokenRevokedError()

        await centralized_cache_service.set_token_data_cached(token, token_data)
        return token_data

    async def authenticate_user(self, db: AsyncSession, email: str, password: str) -> User:
        """
        Authenticate a user with timing attack protection.
        """
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if not user or not user.hashed_password:
            await run_in_threadpool(self.hasher.verify_password, password, self.hasher.DUMMY_HASH)
            raise InvalidCredentialsError()

        password_matches = await run_in_threadpool(
            self.hasher.verify_password, password, user.hashed_password
        )
        if not password_matches:
            raise InvalidCredentialsError()

        if self.hasher.needs_rehash(user.hashed_password):
            user.hashed_password = self.hasher.hash_password(password)

        return user

    async def authenticate_client(
        self, db: AsyncSession, client_id: str, client_secret: str
    ) -> OAuth2Client:
        """Authenticate a confidential OAuth2 client."""
        result = await db.execute(select(OAuth2Client).where(OAuth2Client.client_id == client_id))
        client = result.scalar_one_or_none()

        if not client or client.client_secret != client_secret:
            raise InvalidCredentialsError("Invalid client credentials")

        return client

    def create_client_credentials_token(self, client: OAuth2Client, scopes: list[str]) -> TokenPair:
        """Create a token for Client Credentials flow."""
        allowed_scopes = set(client.scopes or [])
        requested_scopes = set(scopes)

        if not requested_scopes.issubset(allowed_scopes):
            raise InsufficientPermissionsError("Invalid scope requested")

        return self.tokens.create_client_credentials_token(client, scopes)

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

    def verify_mtls(self, request: Request) -> bool:
        """
        Verify mTLS headers from trusted proxies.
        """
        client_verify = request.headers.get("X-SSL-Client-Verify")
        if client_verify != "SUCCESS":
            logger.warning("mtls_verification_failed", status=client_verify)
            return False

        return True


auth_service = AuthService()


def get_auth_service() -> AuthService:
    return auth_service


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

    try:
        token_data = await service.validate_token(token)
    except Exception as e:
        if isinstance(e, TokenRevokedError):
            raise HTTPException(status_code=401, detail="Token revoked")
        # Fallback for TokenExpiredError, InvalidTokenError etc if they are raised directly
        raise HTTPException(status_code=401, detail=str(e))

    user_id = token_data.user_id

    cached_user_data = await centralized_cache_service.get_user_cached(user_id)
    if cached_user_data:
        user = User(**cached_user_data)
        request.state.user = user
        return user

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

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


class RoleRegistry:
    """Config-driven dynamic Registry for Role-based Access Control (RBAC)."""
    
    _ROLES: dict[str, list[str]] = settings.rbac_roles if hasattr(settings, "rbac_roles") else {
        "free": ["free"],
        "pro": ["free", "pro"],
        "enterprise": ["free", "pro", "enterprise", "admin"],
    }

    @classmethod
    def get_effective_roles(cls, tier: str) -> list[str]:
        return cls._ROLES.get(tier.lower(), ["free"])

    @classmethod
    def has_permission(cls, user_tier: str, required_role: str) -> bool:
        return required_role in cls.get_effective_roles(user_tier)

class RoleChecker:
    """Efficient role verification middleware."""
    def __init__(self, required_role: str):
        self.required_role = required_role

    async def __call__(self, user: User = Depends(get_current_active_user)):
        if not RoleRegistry.has_permission(user.tier, self.required_role):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user


async def get_api_key(
    request: Request,
    api_key: str | None = Depends(api_key_header),
    db: AsyncSession = Depends(get_async_db),
) -> User | None:
    """Optimized API Key verification with dual-tier caching."""
    if not api_key:
        return None

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    cached_data = await centralized_cache_service.get_api_key_cached(key_hash)
    
    if cached_data:
        await centralized_cache_service.update_api_key_last_used(key_hash)
        return User(**cached_data)

    from sqlalchemy.orm import joinedload
    result = await db.execute(
        select(APIKey).options(joinedload(APIKey.user)).where(APIKey.key_hash == key_hash, APIKey.is_active)
    )
    record = result.scalar_one_or_none()
    
    if not record:
        return None

    user = record.user
    user_data = {
        "id": str(user.id),
        "email": user.email,
        "tier": user.tier,
        "is_active": user.is_active,
        "is_verified": user.is_verified,
        "mfa_enabled": user.mfa_enabled,
        "key_name": record.name,
    }

    await centralized_cache_service.set_api_key_cached(key_hash, user_data)
    await centralized_cache_service.update_api_key_last_used(key_hash)
    return User(**user_data)


async def get_current_user_flexible(
    request: Request,
    token: str | None = Depends(get_token_from_header),
    api_key_user: User | None = Depends(get_api_key),
    db: AsyncSession = Depends(get_async_db),
    service: AuthService = Depends(get_auth_service),
) -> User | None:
    if api_key_user:
        return api_key_user

    if not token:
        return None

    try:
        return await get_current_user(request, token, db, service)
    except HTTPException:
        return None