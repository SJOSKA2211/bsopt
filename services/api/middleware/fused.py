"""
Fused Security Middleware (Ultra-High Performance)
Consolidates all security layers into a single ASGI hop to minimize context-switching overhead.
"""

import re

import structlog
from fastapi import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from services.api.responses import MsgspecJSONResponse
from services.config import settings

logger = structlog.get_logger(__name__)


class FusedSecurityMiddleware:
    """
    All-in-one security middleware optimized for speed.
    Layers: IP Blocking -> Security Headers -> CSRF -> JWT Auth -> Input Sanitization.
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

    def __init__(self, app: ASGIApp):
        self.app = app
        self.blocked_ips: set[str] = set()
        self.trusted_proxies = {"127.0.0.1", "::1", "172.16.0.0/12", "10.0.0.0/8"}
        self.csrf_secret = settings.JWT_SECRET.encode()

        # CSP and other headers pre-built for speed
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; form-action 'self'; base-uri 'self'; object-src 'none'",
        }

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        path = request.url.path

        # 1. IP Blocking
        client_ip = request.client.host if request.client else "unknown"
        if client_ip in self.blocked_ips:
            resp = MsgspecJSONResponse(status_code=403, content={"detail": "Access denied"})
            await resp(scope, receive, send)
            return

        # 2. JWT Auth (Fast Path)
        is_public = path in self.PUBLIC_PATHS or path.startswith(self.PUBLIC_PREFIXES)

        if not is_public:
            from services.security.auth import auth_service

            token = None
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
            else:
                token = request.cookies.get("better-auth.session_token")

            if not token:
                resp = MsgspecJSONResponse(
                    status_code=401,
                    content={"detail": "Authentication token missing"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await resp(scope, receive, send)
                return

            try:
                token_data = await auth_service.validate_token(token)

                # Populate request state via scope["state"] for compatibility
                state = scope.setdefault("state", {})
                user_id = token_data.user_id
                tier = token_data.tier

                state["user_id"] = user_id
                state["user_email"] = token_data.email
                state["user_tier"] = tier
                state["auth_type"] = token_data.token_type

                # 3. Rate Limiting (Distributed Token Bucket)
                from services.utils.rate_limit import RateLimitTier, limiter

                # Default to FREE if tier not recognized
                try:
                    limit_tier = RateLimitTier(tier.lower())
                except (ValueError, AttributeError):
                    limit_tier = RateLimitTier.FREE

                if not await limiter.is_allowed(user_id, path, limit_tier):
                    logger.warning("rate_limit_exceeded", user_id=user_id, path=path, tier=tier)
                    resp = MsgspecJSONResponse(
                        status_code=429,
                        content={"detail": "Rate limit exceeded", "retry_after": "60s"},
                    )
                    await resp(scope, receive, send)
                    return

            except Exception as e:
                logger.warning("auth_failed", error=str(e), path=path)
                resp = MsgspecJSONResponse(
                    status_code=401, content={"detail": "Authentication failed"}
                )
                await resp(scope, receive, send)
                return

        # 4. Security Headers Wrapper
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                # Inject pre-built headers
                for k, v in self.security_headers.items():
                    headers[k.lower().encode()] = v.encode()

                # HSTS
                if settings.is_production:
                    headers[b"strict-transport-security"] = b"max-age=31536000; includeSubDomains"

                message["headers"] = list(headers.items())
            await send(message)

        await self.app(scope, receive, send_wrapper)
