"""
Zero-Trust Security Middleware (Ultra-High Performance)
Consolidates all security layers into a single ASGI hop to minimize context-switching overhead.
"""

import re

import structlog
from fastapi import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from services.api.responses import MsgspecJSONResponse
from src.config import settings
from src.shared.security import SecurityContext, is_trusted_proxy

logger = structlog.get_logger(__name__)


class ZeroTrustMiddleware:
    """
    All-in-one security middleware optimized for speed.
    Layers: IP Blocking -> Proxy Trust -> JWT Auth -> mTLS (Internal) -> Rate Limiting.
    """

    # Compiled regex for high-performance pattern matching
    DANGEROUS_PATTERN_RE = re.compile(
        r"(<script|javascript:|onclick|onerror|onload|eval\(|document\.cookie|window\.location)",
        re.IGNORECASE,
    )

    # Public path exemptions for Auth
    PUBLIC_PATHS = {
        "/",
        "/health",
        "/api/v1/health",
        "/api/diagnostics/imports",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/graphql",
        "/metrics",
    }

    PUBLIC_PREFIXES = (
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/auth/oauth",
        "/api/v1/auth/.well-known",
        "/api/v1/pricing",
    )

    INTERNAL_PREFIXES = (
        "/api/internal/",
        "/admin/",
    )

    def __init__(self, app: ASGIApp):
        self.app = app
        self.blocked_ips: set[str] = set()
        self.csrf_secret = settings.JWT_SECRET.encode()

        # CSP and other headers pre-built for speed (Binary encoded bounds)
        self.security_headers = [
            (b"x-content-type-options", b"nosniff"),
            (b"x-frame-options", b"DENY"),
            (b"x-xss-protection", b"1; mode=block"),
            (b"referrer-policy", b"strict-origin-when-cross-origin"),
            (b"permissions-policy", b"accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()"),
            (b"content-security-policy", b"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; form-action 'self'; base-uri 'self'; object-src 'none'"),
        ]
        
        if settings.is_production:
            self.security_headers.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))


    async def _handle_auth(self, request: Request, path: str, is_trusted: bool, ssl_verify: str | None, ssl_dn: str | None) -> SecurityContext:
        """Dedicated auth layer for Zero-Trust validation."""
        is_public = path in self.PUBLIC_PATHS or path.startswith(self.PUBLIC_PREFIXES)
        is_internal = path.startswith(self.INTERNAL_PREFIXES)

        # Initialize SecurityContext
        security_context = SecurityContext(
            is_internal=is_internal, service_id=ssl_dn if is_trusted else None
        )

        if is_public:
            return security_context

        from src.auth.auth import auth_service

        token = request.headers.get("Authorization")
        if token and token.startswith("Bearer "):
            token = token.split(" ")[1]
        else:
            token = request.cookies.get("better-auth.session_token")

        if not token:
            raise HTTPException(status_code=401, detail="Authentication token missing")

        # 3. Dedicated Auth Service Hop
        token_data = await auth_service.validate_token(token)

        # 4. Internal Path Cryptographic Certainty
        if is_internal and ssl_verify != "SUCCESS":
            logger.warning("internal_access_denied_no_mtls", path=path)
            raise HTTPException(status_code=403, detail="Internal access requires valid client certificate.")

        # 5. Populate SecurityContext
        security_context.user_id = token_data.user_id
        security_context.email = token_data.email
        security_context.tier = token_data.tier
        security_context.auth_type = token_data.token_type

        # 6. Zero-Trust Rate Limiting
        from src.shared.utils.rate_limit import RateLimitTier, limiter

        tier_str = (token_data.tier or "FREE").upper()
        limit_tier = getattr(RateLimitTier, tier_str, RateLimitTier.FREE)

        if not await limiter.is_allowed(token_data.user_id, path, limit_tier):
            logger.warning("rate_limit_exceeded", user_id=token_data.user_id, path=path, tier=tier_str)
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Upgrade tier for higher limits.",
                headers={"X-RateLimit-Limit": str(limit_tier.value)},
            )

        return security_context

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        path = request.url.path

        # 1. IP Blocking & Proxy Trust
        client_ip = request.client.host if request.client else "unknown"
        if client_ip in self.blocked_ips:
            resp = MsgspecJSONResponse(status_code=403, content={"detail": "Access denied"})
            await resp(scope, receive, send)
            return

        # Verify against TRUSTED_PROXIES
        is_trusted = is_trusted_proxy(client_ip, settings.TRUSTED_PROXIES)
        ssl_verify = request.headers.get("X-SSL-Client-Verify") if is_trusted else None
        ssl_dn = request.headers.get("X-SSL-Client-S-DN") if is_trusted else None

        try:
            security_context = await self._handle_auth(request, path, is_trusted, ssl_verify, ssl_dn)
            
            # Attach to scope state
            state = scope.setdefault("state", {})
            state["security_context"] = security_context

            # Legacy support for existing code
            if security_context.user_id:
                state["user_id"] = security_context.user_id
                state["user_email"] = security_context.email
                state["user_tier"] = security_context.tier
                state["auth_type"] = security_context.auth_type

        except HTTPException as e:
            resp = MsgspecJSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail},
                headers=e.headers,
            )
            await resp(scope, receive, send)
            return
        except Exception as e:
            logger.warning("zero_trust_auth_intercept_failed", error=str(e), path=path)
            resp = MsgspecJSONResponse(
                status_code=401,
                content={"detail": "Unauthorized: Zero-Trust validation failed."},
            )
            await resp(scope, receive, send)
            return

        # 7. Security Headers Wrapper (High-Performance Zero-Copy Tuple Extend)
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = message.get("headers", [])
                headers.extend(self.security_headers)
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)
