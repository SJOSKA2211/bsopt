"""
JWT Validator Middleware for FastAPI

Unified JWT validation supporting:
- RS256 (RSA Signature with SHA-256)
- ES256 (ECDSA with P-256 and SHA-256)
- HS256 (HMAC with SHA-256)

Features:
- Redis caching for validated tokens (TTL: 5 minutes)
- Token blacklist checking
- Automatic key selection based on algorithm
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redis.asyncio import Redis

import jwt
import structlog
from fastapi import HTTPException, Request, Response
from jwt.exceptions import (
    ExpiredSignatureError,
    InvalidTokenError,
    PyJWTError,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.shared.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class JWTClaims:
    """Validated JWT claims."""

    sub: str
    email: str | None = None
    tier: str = "free"
    roles: list[str] = None
    exp: int = 0
    iat: int = 0
    jti: str | None = None
    token_type: str = "access"
    issuer: str | None = None
    audience: str | None = None

    def __post_init__(self):
        if self.roles is None:
            self.roles = []


class JWTValidator:
    """
    High-performance JWT validator with caching and blacklist support.
    """

    def __init__(
        self,
        redis_client: Redis | None = None,
        cache_ttl: int = 300,
        blacklist_ttl: int = 3600,
    ):
        self.redis = redis_client
        self.cache_ttl = cache_ttl
        self.blacklist_ttl = blacklist_ttl

    def _get_cache_key(self, token: str) -> str:
        """Generate cache key for token."""
        token_hash = hashlib.sha256(token.encode()).hexdigest()[:32]
        return f"jwt:validated:{token_hash}"

    async def _get_from_cache(self, token: str) -> JWTClaims | None:
        """Retrieve validated claims from cache."""
        if not self.redis:
            return None

        try:
            cache_key = self._get_cache_key(token)
            cached = await self.redis.get(cache_key)
            if cached:
                data = json.loads(cached)
                return JWTClaims(**data)
        except Exception as e:
            logger.warning("jwt_cache_read_failed", error=str(e))

        return None

    async def _cache_claims(self, token: str, claims: JWTClaims) -> None:
        """Cache validated claims."""
        if not self.redis:
            return

        try:
            cache_key = self._get_cache_key(token)
            await self.redis.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(
                    {
                        "sub": claims.sub,
                        "email": claims.email,
                        "tier": claims.tier,
                        "roles": claims.roles,
                        "exp": claims.exp,
                        "iat": claims.iat,
                        "jti": claims.jti,
                        "token_type": claims.token_type,
                        "issuer": claims.issuer,
                        "audience": claims.audience,
                    }
                ),
            )
        except Exception as e:
            logger.warning("jwt_cache_write_failed", error=str(e))

    async def _is_blacklisted(self, jti: str) -> bool:
        """Check if token JTI is blacklisted."""
        if not self.redis:
            return False

        try:
            blacklist_key = f"jwt:blacklist:{jti}"
            return await self.redis.exists(blacklist_key)
        except Exception as e:
            logger.warning("jwt_blacklist_check_failed", error=str(e))
            return False

    async def blacklist_token(self, jti: str, exp: int) -> None:
        """Add token to blacklist."""
        if not self.redis:
            return

        try:
            blacklist_key = f"jwt:blacklist:{jti}"
            ttl = max(exp - int(time.time()), 0)
            if ttl > 0:
                await self.redis.setex(blacklist_key, ttl, "1")
                logger.info("token_blacklisted", jti=jti, ttl=ttl)
        except Exception as e:
            logger.error("jwt_blacklist_failed", error=str(e))

    def _get_private_key(self, algorithm: str) -> str:
        """Get private key for signing based on algorithm."""
        if algorithm.startswith("RS"):
            return settings.rsa_private_key
        elif algorithm.startswith("ES"):
            return settings.es256_private_key
        elif algorithm == "HS256":
            return settings.JWT_SECRET
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    def _get_public_key(self, algorithm: str) -> str:
        """Get public key for verification based on algorithm."""
        if algorithm.startswith("RS"):
            return settings.rsa_public_key
        elif algorithm.startswith("ES"):
            return settings.es256_public_key
        elif algorithm == "HS256":
            return settings.JWT_SECRET
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    async def validate(self, token: str) -> JWTClaims:
        """
        Validate JWT token and return claims.

        Steps:
        1. Check Redis cache
        2. Decode and verify signature
        3. Check blacklist
        4. Cache valid result
        """
        cached_claims = await self._get_from_cache(token)
        if cached_claims:
            logger.debug("jwt_cache_hit", sub=cached_claims.sub)
            return cached_claims

        try:
            # 🛡️ Sentinel: Enforce expected algorithm to prevent algorithm confusion attacks
            algorithm = settings.JWT_ALGORITHM
            public_key = self._get_public_key(algorithm)

            payload = jwt.decode(
                token,
                public_key,
                algorithms=[algorithm],
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_aud": False,
                    "require": ["exp", "sub"],
                },
            )

            jti = payload.get("jti")
            if jti and await self._is_blacklisted(jti):
                logger.warning("jwt_blacklisted_token", jti=jti)
                raise HTTPException(status_code=401, detail="Token has been revoked")

            claims = JWTClaims(
                sub=payload.get("sub"),
                email=payload.get("email"),
                tier=payload.get("tier", "free"),
                roles=payload.get("roles", []),
                exp=payload.get("exp", 0),
                iat=payload.get("iat", 0),
                jti=jti,
                token_type=payload.get("type", "access"),
                issuer=payload.get("iss"),
                audience=payload.get("aud"),
            )

            await self._cache_claims(token, claims)
            logger.debug("jwt_validated", sub=claims.sub, algorithm=algorithm)

            return claims

        except ExpiredSignatureError:
            logger.warning("jwt_expired")
            raise HTTPException(status_code=401, detail="Token has expired")
        except InvalidTokenError as e:
            logger.warning("jwt_invalid", error=str(e))
            raise HTTPException(status_code=401, detail="Invalid token")
        except PyJWTError as e:
            logger.error("jwt_error", error=str(e))
            raise HTTPException(status_code=401, detail="Token validation failed")

    def create_token(
        self,
        user_id: str,
        email: str,
        tier: str = "free",
        roles: list[str] = None,
        token_type: str = "access",
        additional_claims: dict[str, object] = None,
    ) -> tuple[str, int]:
        """
        Create a new JWT token.

        Returns:
            Tuple of (token, expires_in_seconds)
        """
        import secrets
        from datetime import timedelta

        if roles is None:
            roles = []

        if token_type == "access":
            expires_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        else:
            expires_delta = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

        now = datetime.now(UTC)
        exp = now + expires_delta

        payload = {
            "sub": user_id,
            "email": email,
            "tier": tier,
            "roles": roles,
            "type": token_type,
            "iat": int(now.timestamp()),
            "exp": int(exp.timestamp()),
            "jti": secrets.token_hex(16),
        }

        if additional_claims:
            payload.update(additional_claims)

        algorithm = settings.JWT_ALGORITHM
        private_key = self._get_private_key(algorithm)

        token = jwt.encode(payload, private_key, algorithm=algorithm)

        return token, int(expires_delta.total_seconds())


class JWTValidatorMiddleware(BaseHTTPMiddleware):
    """
    ASGI Middleware for JWT validation.

    Validates JWT tokens on protected routes and attaches user info to request state.
    """

    EXCLUDED_PATHS = {
        "/health",
        "/ready",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/graphql",
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/auth/refresh",
    }

    def __init__(
        self,
        app: ASGIApp,
        redis_client: Redis | None = None,
        excluded_paths: set[str] | None = None,
    ):
        super().__init__(app)
        self.validator = JWTValidator(redis_client=redis_client)
        self.excluded_paths = excluded_paths or self.EXCLUDED_PATHS

    def _should_skip(self, path: str) -> bool:
        """Check if path should skip JWT validation."""
        if path in self.excluded_paths:
            return True

        for excluded in self.excluded_paths:
            if path.startswith(excluded):
                return True

        return False

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with JWT validation."""
        if self._should_skip(request.url.path):
            return await call_next(request)

        auth_header = request.headers.get("Authorization")

        if not auth_header:
            request.state.user = None
            request.state.jwt_claims = None
            return await call_next(request)

        if not auth_header.startswith("Bearer "):
            return await call_next(request)

        token = auth_header[7:]

        try:
            claims = await self.validator.validate(token)

            request.state.user = {
                "user_id": claims.sub,
                "email": claims.email,
                "tier": claims.tier,
                "roles": claims.roles,
            }
            request.state.jwt_claims = claims
            request.state.token_jti = claims.jti

        except HTTPException:
            raise
        except Exception as e:
            logger.error("jwt_middleware_error", error=str(e))
            raise HTTPException(status_code=500, detail="Authentication error")

        return await call_next(request)


def get_jwt_validator() -> JWTValidator:
    """Get JWT validator instance."""
    return JWTValidator()


async def require_auth(request: Request) -> JWTClaims:
    """FastAPI dependency for requiring authentication."""
    claims = getattr(request.state, "jwt_claims", None)

    if not claims:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return claims


async def require_tier(request: Request, allowed_tiers: list[str]) -> JWTClaims:
    """FastAPI dependency for tier-based access control."""
    claims = await require_auth(request)

    if claims.tier not in allowed_tiers:
        raise HTTPException(
            status_code=403,
            detail="Insufficient subscription tier",
        )

    return claims


async def require_role(request: Request, required_roles: list[str]) -> JWTClaims:
    """FastAPI dependency for role-based access control."""
    claims = await require_auth(request)

    if not any(role in claims.roles for role in required_roles):
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions",
        )

    return claims
