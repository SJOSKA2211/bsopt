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

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from redis.asyncio import Redis

import structlog
from fastapi import Depends, HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.shared.config import settings

logger = structlog.get_logger(__name__)

from src.auth.core.sessions import session_service
from src.auth.core.tokens import TokenData as JWTClaims
from src.auth.core.tokens import token_service


class JWTValidator:
    """
    High-performance JWT validator delegating to TokenService and SessionService.
    """

    def __init__(
        self,
        redis_client: Any = None,
        cache_ttl: int = 300,
        blacklist_ttl: int = 3600,
    ):
        self.tokens = token_service
        self.sessions = session_service

    async def validate(self, token: str) -> JWTClaims:
        """
        Validate JWT token and return claims.
        Fast-path Redis sync included.
        """
        # 1. Fast Path (Redis)
        cached = await self.sessions.get_cached_session(token)
        if cached:
            return cached

        # 2. Signature & Revocation Check
        token_data = self.tokens.decode_token(token)
        if token_data.jti and await self.sessions.is_token_revoked(token_data.jti):
            raise HTTPException(status_code=401, detail="Token revoked")

        # 3. Cache valid result
        await self.sessions.cache_session(token, token_data)

        return token_data

    def create_token(
        self,
        user_id: str,
        email: str,
        tier: str = "free",
        roles: list[str] = None,
        token_type: str = "access",
        additional_claims: dict[str, object] = None,
    ) -> tuple[str, int]:
        """Create a new JWT token via centralized TokenService."""
        token_pair = self.tokens.create_token_pair(user_id, email, tier, scopes=roles or [])
        token = token_pair.access_token if token_type == "access" else token_pair.refresh_token
        return token, settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60


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
                "user_id": claims.user_id,
                "email": claims.email,
                "tier": claims.tier,
                "roles": claims.scopes,
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


def require_tier(allowed_tiers: list[str]):
    """FastAPI dependency factory for tier-based access control."""

    async def _require_tier(claims: JWTClaims = Depends(require_auth)) -> JWTClaims:
        if claims.tier not in allowed_tiers:
            raise HTTPException(
                status_code=403,
                detail="Insufficient subscription tier",
            )

        return claims

    return _require_tier


def require_role(required_roles: list[str]):
    """FastAPI dependency factory for role-based access control."""

    async def _require_role(claims: JWTClaims = Depends(require_auth)) -> JWTClaims:
        if not any(role in claims.roles for role in required_roles):
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions",
            )

        return claims

    return _require_role