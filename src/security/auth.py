"""
JWT Authentication Service
==========================

Secure JWT-based authentication with:
- Access and refresh token support
- Token blacklisting for logout
- Role-based access control
- FastAPI dependency injection
"""

import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union, cast
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, APIKeyHeader
import hashlib
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from src.config import settings
from src.database import get_db
from src.database.models import User

from .password import password_service

logger = logging.getLogger(__name__)

# Security schemes for FastAPI docs
security_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass
class TokenData:
    """Decoded token data."""

    user_id: str
    email: str
    tier: str
    token_type: str  # "access" or "refresh"
    exp: datetime
    iat: datetime
    jti: Optional[str] = None  # JWT ID for blacklisting


@dataclass
class TokenPair:
    """Access and refresh token pair."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 0  # Seconds until access token expires


class TokenBlacklist:
    """
    Token blacklist for invalidated tokens.

    In production, use Redis for distributed blacklist.
    """

    def __init__(self) -> None:
        self._blacklist: set = set()
        self._redis = None

    async def initialize(self, redis_client=None):
        """Initialize with optional Redis client."""
        self._redis = redis_client

    async def add(self, jti: str, exp: datetime) -> None:
        """Add token to blacklist."""
        if self._redis:
            # Store with TTL matching token expiration
            ttl = int((exp - datetime.now(timezone.utc)).total_seconds())
            if ttl > 0:
                await self._redis.setex(f"blacklist:{jti}", ttl, "1")
        else:
            self._blacklist.add(jti)

    async def contains(self, jti: str) -> bool:
        """Check if token is blacklisted."""
        if self._redis:
            return bool(await self._redis.exists(f"blacklist:{jti}"))
        return jti in self._blacklist

    async def cleanup(self) -> int:
        """Remove expired tokens from in-memory blacklist."""
        count = len(self._blacklist)
        self._blacklist.clear()
        return count


# Global token blacklist
token_blacklist = TokenBlacklist()


# FastAPI Dependencies


async def get_token_from_header(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
) -> Optional[str]:
    """Extract token from Authorization header."""
    if credentials:
        return credentials.credentials
    return None


class AuthService:
    """
    Authentication service for JWT token management.

    Handles token creation, validation, and user authentication.
    """

    @property
    def private_key(self):
        return settings.JWT_PRIVATE_KEY

    @property
    def public_key(self):
        return settings.JWT_PUBLIC_KEY

    @property
    def algorithm(self):
        return settings.JWT_ALGORITHM

    def __init__(self) -> None:
        self.access_token_expire = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        self.refresh_token_expire = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    def create_access_token(
        self,
        user_id: Union[str, UUID],
        email: str,
        tier: str,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a JWT access token."""
        now = datetime.now(timezone.utc)
        expire = now + self.access_token_expire

        payload = {
            "sub": str(user_id),
            "email": email,
            "tier": tier,
            "type": "access",
            "iat": int(now.timestamp()),
            "exp": int(expire.timestamp()),
            "jti": self._generate_jti(),
        }

        if additional_claims:
            payload.update(additional_claims)

        return cast(str, jwt.encode(payload, self.private_key, algorithm=self.algorithm))

    def create_refresh_token(
        self,
        user_id: Union[str, UUID],
        email: str,
    ) -> str:
        """Create a JWT refresh token."""
        now = datetime.now(timezone.utc)
        expire = now + self.refresh_token_expire

        payload = {
            "sub": str(user_id),
            "email": email,
            "type": "refresh",
            "iat": int(now.timestamp()),
            "exp": int(expire.timestamp()),
            "jti": self._generate_jti(),
        }

        return cast(str, jwt.encode(payload, self.private_key, algorithm=self.algorithm))

    def create_token_pair(
        self,
        user_id: Union[str, UUID],
        email: str,
        tier: str,
    ) -> TokenPair:
        """Create both access and refresh tokens."""
        access_token = self.create_access_token(user_id, email, tier)
        refresh_token = self.create_refresh_token(user_id, email)

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=int(self.access_token_expire.total_seconds()),
        )

    def decode_token(self, token: str) -> TokenData:
        """Decode and validate a JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=[self.algorithm],
            )

            return TokenData(
                user_id=payload.get("sub"),
                email=payload.get("email", ""),
                tier=payload.get("tier", "free"),
                token_type=payload.get("type", "access"),
                exp=datetime.fromtimestamp(payload.get("exp", 0), tz=timezone.utc),
                iat=datetime.fromtimestamp(payload.get("iat", 0), tz=timezone.utc),
                jti=payload.get("jti"),
            )

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except JWTError as e:
            logger.warning(f"JWT decode error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def validate_token(
        self, token: Optional[str] = Depends(get_token_from_header)
    ) -> TokenData:
        """Validate token including blacklist check."""
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token_data = self.decode_token(token)

        if token_data.jti and await token_blacklist.contains(token_data.jti):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return token_data

    async def invalidate_token(self, token: str, request: Optional[Request] = None) -> None:
        """Invalidate a token by adding to blacklist."""
        try:
            token_data = self.decode_token(token)
            if token_data.jti:
                await token_blacklist.add(token_data.jti, token_data.exp)
        except HTTPException:
            pass

    def authenticate_user(
        self,
        db: Session,
        email: str,
        password: str,
        request: Optional[Request] = None,
    ) -> Optional[User]:
        """Authenticate user with email and password."""
        from .audit import AuditEvent, log_audit

        user = db.query(User).filter(User.email == email).first()

        if not user:
            password_service.verify_password(
                password, "$2b$12$invalid.hash.for.timing.attack.prevention"
            )
            log_audit(
                AuditEvent.USER_LOGIN_FAILURE,
                request=request,
                details={"email": email, "reason": "user_not_found"},
                persist_to_db=False,
            )
            return None

        if not password_service.verify_password(password, user.hashed_password):
            log_audit(
                AuditEvent.USER_LOGIN_FAILURE,
                user=user,
                request=request,
                details={"reason": "invalid_password"},
            )
            return None

        if not user.is_active:
            log_audit(
                AuditEvent.USER_LOGIN_FAILURE,
                user=user,
                request=request,
                details={"reason": "user_inactive"},
            )
            return None

        log_audit(AuditEvent.USER_LOGIN_SUCCESS, user=user, request=request)
        return user

    @staticmethod
    def _generate_jti() -> str:
        """Generate unique JWT ID."""
        return secrets.token_urlsafe(16)


# Global auth service instance
auth_service = AuthService()

def get_auth_service() -> AuthService:
    """Return the global auth service instance."""
    return auth_service


# FastAPI Dependencies


async def get_current_user(
    request: Request,
    token: Optional[str] = Depends(get_token_from_header),
    db: Session = Depends(get_db),
) -> User:
    """FastAPI dependency to get current authenticated user."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

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
        logger.warning(f"Failed to get user from cache: {e}")

    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        from src.utils.cache import db_cache

        user_data_dict = {c.name: getattr(user, c.name) for c in user.__table__.columns}
        await db_cache.set_user(user_id, user_data_dict)
    except Exception as e:
        logger.warning(f"Failed to set user in cache: {e}")

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


async def get_optional_user(
    token: Optional[str] = Depends(get_token_from_header),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Get current user if authenticated, None otherwise."""
    if not token:
        return None

    try:
        token_data = await auth_service.validate_token(token)
        return db.query(User).filter(User.id == token_data.user_id).first()
    except HTTPException:
        return None


def require_tier(allowed_tiers: List[str]):
    """Factory for tier-based access control dependency."""

    async def check_tier(user: User = Depends(get_current_active_user)) -> User:
        if user.tier not in allowed_tiers:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This endpoint requires one of: {', '.join(allowed_tiers)} tier",
            )
        return user

    return check_tier


def require_admin():
    """Dependency for admin-only endpoints."""
    return require_tier(["enterprise"])


async def get_api_key(
    request: Request,
    api_key: Optional[str] = Depends(api_key_header),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Validate API key and return associated user."""
    if not api_key:
        return None
    
    # Hash the key to check against DB
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    
    from sqlalchemy import and_
    from src.database.models import APIKey
    key_record = db.query(APIKey).filter(
        and_(APIKey.key_hash == key_hash, APIKey.is_active == True)
    ).first()
    
    if not key_record:
        logger.warning(f"Invalid API Key attempt with prefix: {api_key[:8]}")
        return None
        
    # Update last used
    key_record.last_used_at = datetime.now(timezone.utc)
    try:
        db.commit()
    except Exception as e:
        logger.error(f"Failed to update API key last_used_at: {e}")
    
    user = key_record.user
    request.state.user = user
    return user


async def get_current_user_flexible(


    request: Request,


    token: Optional[str] = Depends(get_token_from_header),


    api_key_user: Optional[User] = Depends(get_api_key),


    db: Session = Depends(get_db),


) -> User:


    """Authentication via either JWT or API Key."""


    if api_key_user:


        if not api_key_user.is_active:


             raise HTTPException(


                status_code=status.HTTP_403_FORBIDDEN,


                detail="Account associated with API key is disabled",


            )


        return api_key_user


        


    if not token:


        raise HTTPException(


            status_code=status.HTTP_401_UNAUTHORIZED,


            detail="Not authenticated. Provide Bearer token or X-API-Key header.",


            headers={"WWW-Authenticate": "Bearer"},


        )


        


    return await get_current_user(request, token, db)

