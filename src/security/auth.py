"""
JWT Authentication Service (Async)
==================================

Secure JWT-based authentication with:
- Async database operations using AsyncSession
- Access and refresh token support
- Token blacklisting for logout
- FastAPI dependency injection
"""

import hashlib
import inspect
import logging
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jwt.exceptions import ExpiredSignatureError, PyJWTError
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from starlette.concurrency import run_in_threadpool

from src.config import settings
from src.database import get_async_db
from src.database.models import APIKey, User

from .password import password_service

logger = logging.getLogger(__name__)

# Security schemes for FastAPI docs
security_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass
class TokenData:
    user_id: str
    email: str
    tier: str
    token_type: str
    exp: datetime
    iat: datetime
    jti: str | None = None


@dataclass
class TokenPair:
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 0


class TokenBlacklist:
    def __init__(self) -> None:
        self._blacklist: set = set()
        self._redis = None

    async def initialize(self, redis_client=None):
        self._redis = redis_client

    async def add(self, jti: str, exp: datetime) -> None:
        if self._redis:
            ttl = int((exp - datetime.now(UTC)).total_seconds())
            if ttl > 0:
                await self._redis.setex(f"blacklist:{jti}", ttl, "1")
        else:
            self._blacklist.add(jti)

    async def contains(self, jti: str) -> bool:
        if self._redis:
            return bool(await self._redis.exists(f"blacklist:{jti}"))
        return jti in self._blacklist

token_blacklist = TokenBlacklist()

async def get_token_from_header(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
) -> str | None:
    if credentials:
        return credentials.credentials
    return None


class AuthService:
    @property
    def private_key(self): return settings.JWT_PRIVATE_KEY
    @property
    def public_key(self): return settings.JWT_PUBLIC_KEY
    @property
    def algorithm(self): return settings.JWT_ALGORITHM
    @property
    def access_token_expire(self): return timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    @property
    def refresh_token_expire(self): return timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    async def authenticate_user(
        self, db: Any, email: str, password: str, request: Request
    ) -> User | None:
        """Authenticate a user by email and password with timing attack protection."""
        
        # Check if db.execute is an async function (AsyncSession)
        is_async = hasattr(db, 'execute') and inspect.iscoroutinefunction(db.execute)
        
        if is_async: 
             result = await db.execute(select(User).where(User.email == email))
             user = result.scalar_one_or_none()
        else: # Fallback for sync session or MagicMock
             user = db.query(User).filter(User.email == email).first()
        
        # Timing attack protection: always verify a password hash, even if the user is not found.
        # This prevents attackers from enumerating valid usernames based on response times.
        user_exists = True
        if not user:
            user_exists = False
            # Create a dummy user with a placeholder hashed password to ensure password verification always runs
            user = User(
                email="nonexistent@example.com",
                hashed_password=password_service.hash_password(secrets.token_urlsafe(32)), # Use a random hash
                full_name="Dummy User",
                tier="free",
                is_active=False,
                is_verified=False,
                created_at=datetime.now(UTC)
            )
        
        # Always run password verification
        password_matches = await run_in_threadpool(password_service.verify_password, password, user.hashed_password)

        if not user_exists or not password_matches:
            return None
            
        # Optimization: Rehash legacy passwords on successful login to migrate to Argon2id
        if password_service.needs_rehash(user.hashed_password):
            logger.info("password_rehash_triggered_on_login", user_id=str(user.id), email=user.email)
            user.hashed_password = await run_in_threadpool(password_service.hash_password, password)
            # Persisted by the commit in the calling login route
            
        return user

    def create_token_pair(self, user_id: str, email: str, tier: str) -> TokenPair:
        """Create a pair of access and refresh tokens."""
        access_token = self._create_token(
            {"sub": user_id, "email": email, "tier": tier, "type": "access"},
            self.access_token_expire,
        )
        refresh_token = self._create_token(
            {"sub": user_id, "email": email, "tier": tier, "type": "refresh"},
            self.refresh_token_expire,
        )
        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=int(self.access_token_expire.total_seconds()),
        )

    def _create_token(self, data: dict, expires_delta: timedelta) -> str:
        """Internal helper to create a JWT token."""
        to_encode = data.copy()
        expire = datetime.now(UTC) + expires_delta
        to_encode.update({"exp": expire, "iat": datetime.now(UTC), "jti": secrets.token_hex(16)})
        return jwt.encode(to_encode, self.private_key, algorithm=self.algorithm)

    async def invalidate_token(self, token: str, request: Request) -> None:
        """Invalidate a token by adding its JTI to the blacklist."""
        try:
            payload = jwt.decode(token, self.public_key, algorithms=[self.algorithm])
            jti = payload.get("jti")
            exp_timestamp = payload.get("exp")
            if jti and exp_timestamp:
                exp = datetime.fromtimestamp(exp_timestamp, tz=UTC)
                await token_blacklist.add(jti, exp)
        except PyJWTError:
            pass

    def decode_token(self, token: str) -> TokenData:
        try:
            payload = jwt.decode(token, self.public_key, algorithms=[self.algorithm])
            return TokenData(
                user_id=payload.get("sub"),
                email=payload.get("email", ""),
                tier=payload.get("tier", "free"),
                token_type=payload.get("type", "access"),
                exp=datetime.fromtimestamp(payload.get("exp", 0), tz=UTC),
                iat=datetime.fromtimestamp(payload.get("iat", 0), tz=UTC),
                jti=payload.get("jti"),
            )
        except ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    async def validate_token(self, token: str | None = Depends(get_token_from_header)) -> TokenData:
        if not token:
            raise HTTPException(status_code=401, detail="Not authenticated")
        token_data = self.decode_token(token)
        if token_data.jti and await token_blacklist.contains(token_data.jti):
            raise HTTPException(status_code=401, detail="Token revoked")
        return token_data

auth_service = AuthService()
def get_auth_service(): return auth_service

async def verify_token_claims_only(
    token: str | None = Depends(get_token_from_header),
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenData:
    """
    Performance-optimized dependency for high-frequency endpoints.
    Verifies JWT signature and claims only. No DB/Cache hits.
    """
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # decode_token already verifies signature, expiration and format
    return auth_service.decode_token(token)

async def get_current_user(
    request: Request,
    token: str | None = Depends(get_token_from_header),
    db: AsyncSession = Depends(get_async_db),
    auth_service: AuthService = Depends(get_auth_service),
) -> User:
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    token_data = await auth_service.validate_token(token)
    user_id = token_data.user_id

    try:
        from src.utils.cache import db_cache
        cached_user_data = await db_cache.get_user(user_id)
        if cached_user_data:
            user = User(**cached_user_data)
            request.state.user = user
            return user
    except Exception as e:
        logger.warning(f"Cache lookup failed: {e}")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    request.state.user = user
    return user

async def get_current_active_user(
    user: User = Depends(get_current_user),
) -> User:
    """Get current user and verify they are active."""
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )
    return user

def require_tier(allowed_tiers: list):
    def decorator(user: User = Depends(get_current_active_user)):
        if user.tier not in allowed_tiers:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient subscription tier",
            )
        return user
    return decorator

async def get_api_key(
    request: Request,
    api_key: str | None = Depends(api_key_header),
    db: AsyncSession = Depends(get_async_db),
) -> User | None:
    if not api_key:
        return None
    
    # Use SHA256 for high-entropy API keys (fast lookup). Not a user password.
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()  # codeql[py/weak-password-hashing]
    
    result = await db.execute(
        select(APIKey).options(selectinload(APIKey.user)).where(
            and_(APIKey.key_hash == key_hash, APIKey.is_active == True)
        )
    )
    key_record = result.scalar_one_or_none()
    
    if not key_record:
        return None
        
    key_record.last_used_at = datetime.now(UTC)
    await db.commit()
    
    request.state.user = key_record.user
    return key_record.user

async def get_current_user_flexible(
    request: Request,
    token: str | None = Depends(get_token_from_header),
    api_key_user: User | None = Depends(get_api_key),
    db: AsyncSession = Depends(get_async_db),
    auth_service: AuthService = Depends(get_auth_service),
) -> User | Any:
    # 1. API Key Auth (Programmatic)
    if api_key_user:
        return api_key_user
        
    # 2. 🚀 SINGULARITY: Better Auth Session (New Primary Auth)
    from src.auth.better_auth import get_current_user as get_better_auth_user
    try:
        # Check if we have a Better Auth session
        better_user = await get_better_auth_user(request, db)
        if better_user:
            return better_user
    except Exception:
        pass # Fallback to legacy JWT if Better Auth fails/missing

    # 3. Legacy JWT Auth (Migration Path)
    try:
        return await get_current_user(request, token, db, auth_service)
    except HTTPException:
        raise
        
    