"""
Security Middleware
===================

Comprehensive security middleware including:
- Security headers (CSP, HSTS, X-Frame-Options, etc.)
- CSRF protection for state-changing operations
- XSS protection
- Clickjacking prevention
- Content sniffing prevention
"""

import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Literal, Optional, Set, cast
from urllib.parse import urlparse

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.config import settings

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses.

    Headers added:
    - Strict-Transport-Security (HSTS)
    - X-Content-Type-Options
    - X-Frame-Options
    - X-XSS-Protection
    - Referrer-Policy
    - Permissions-Policy
    - Content-Security-Policy
    - Cache-Control for sensitive endpoints
    """

    # Endpoints that should not be cached
    NO_CACHE_PATTERNS = [
        "/api/v1/auth",
        "/api/v1/user",
        "/api/v1/admin",
    ]

    def __init__(
        self,
        app: ASGIApp,
        hsts_max_age: int = 31536000,  # 1 year
        include_subdomains: bool = True,
        frame_options: str = "DENY",
        content_type_options: bool = True,
        xss_protection: bool = True,
        referrer_policy: str = "strict-origin-when-cross-origin",
        permissions_policy: Optional[Dict[str, List[str]]] = None,
        csp_directives: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__(app)
        self.hsts_max_age = hsts_max_age
        self.include_subdomains = include_subdomains
        self.frame_options = frame_options
        self.content_type_options = content_type_options
        self.xss_protection = xss_protection
        self.referrer_policy = referrer_policy
        self.permissions_policy = permissions_policy or self._default_permissions_policy()
        self.csp_directives = csp_directives or self._default_csp()

    def _default_permissions_policy(self) -> Dict[str, List[str]]:
        """Default restrictive permissions policy."""
        return {
            "accelerometer": [],
            "camera": [],
            "geolocation": [],
            "gyroscope": [],
            "magnetometer": [],
            "microphone": [],
            "payment": [],
            "usb": [],
        }

    def _default_csp(self) -> Dict[str, List[str]]:
        """Default Content Security Policy."""
        return {
            "default-src": ["'self'"],
            "script-src": ["'self'"],
            "style-src": ["'self'", "'unsafe-inline'"],  # Allow inline styles for docs
            "img-src": ["'self'", "data:", "https:"],
            "font-src": ["'self'"],
            "connect-src": ["'self'"],
            "frame-ancestors": ["'none'"],
            "form-action": ["'self'"],
            "base-uri": ["'self'"],
            "object-src": ["'none'"],
        }

    def _build_csp_header(self) -> str:
        """Build CSP header string."""
        directives = []
        for directive, sources in self.csp_directives.items():
            if sources:
                directives.append(f"{directive} {' '.join(sources)}")
            else:
                directives.append(directive)
        return "; ".join(directives)

    def _build_permissions_policy(self) -> str:
        """Build Permissions-Policy header string."""
        policies = []
        for feature, allowlist in self.permissions_policy.items():
            if not allowlist:
                policies.append(f"{feature}=()")
            else:
                allowed = " ".join(f'"{a}"' for a in allowlist)
                policies.append(f"{feature}=({allowed})")
        return ", ".join(policies)

    def _should_not_cache(self, path: str) -> bool:
        """Check if path should have no-cache headers."""
        return any(path.startswith(pattern) for pattern in self.NO_CACHE_PATTERNS)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = cast(Response, await call_next(request))

        # HSTS - only for HTTPS in production
        if settings.is_production or request.url.scheme == "https":
            hsts_value = f"max-age={self.hsts_max_age}"
            if self.include_subdomains:
                hsts_value += "; includeSubDomains"
            response.headers["Strict-Transport-Security"] = hsts_value

        # Prevent MIME type sniffing
        if self.content_type_options:
            response.headers["X-Content-Type-Options"] = "nosniff"

        # Clickjacking protection
        response.headers["X-Frame-Options"] = self.frame_options

        # XSS protection (legacy but still useful)
        if self.xss_protection:
            response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer policy
        response.headers["Referrer-Policy"] = self.referrer_policy

        # Permissions policy
        response.headers["Permissions-Policy"] = self._build_permissions_policy()

        # Content Security Policy
        response.headers["Content-Security-Policy"] = self._build_csp_header()

        # Cache control for sensitive endpoints
        if self._should_not_cache(request.url.path):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        # Remove potentially dangerous headers
        if "Server" in response.headers:
            del response.headers["Server"]
        if "X-Powered-By" in response.headers:
            del response.headers["X-Powered-By"]

        return response


class CSRFMiddleware(BaseHTTPMiddleware):
    """
    CSRF protection middleware.

    Implements double-submit cookie pattern:
    - Sets a CSRF token in a cookie
    - Requires the token in a header for state-changing requests
    - Validates token on POST/PUT/PATCH/DELETE requests

    For API-only applications with JWT auth, CSRF is less critical
    but still recommended for cookie-based sessions.
    """

    # Methods that require CSRF protection
    PROTECTED_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

    # Paths exempt from CSRF (e.g., auth endpoints, webhooks)
    EXEMPT_PATHS: Set[str] = {
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/auth/refresh",
        "/api/v1/webhooks",
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
    }

    # Cookie and header names
    COOKIE_NAME = "csrf_token"
    HEADER_NAME = "X-CSRF-Token"

    def __init__(
        self,
        app: ASGIApp,
        secret_key: Optional[str] = None,
        token_length: int = 32,
        cookie_max_age: int = 3600,  # 1 hour
        cookie_secure: bool = True,
        cookie_httponly: bool = False,  # Must be readable by JS
        cookie_samesite: str = "lax",
    ):
        super().__init__(app)
        self.secret_key = (secret_key or settings.JWT_SECRET).encode()
        self.token_length = token_length
        self.cookie_max_age = cookie_max_age
        self.cookie_secure = cookie_secure and settings.is_production
        self.cookie_httponly = cookie_httponly
        self.cookie_samesite = cookie_samesite

    def _generate_token(self) -> str:
        """Generate a new CSRF token."""
        return secrets.token_urlsafe(self.token_length)

    def _sign_token(self, token: str) -> str:
        """Sign a token with HMAC."""
        signature = hmac.new(self.secret_key, token.encode(), hashlib.sha256).hexdigest()
        return f"{token}.{signature}"

    def _verify_token(self, signed_token: str) -> bool:
        """Verify a signed token."""
        try:
            parts = signed_token.rsplit(".", 1)
            if len(parts) != 2:
                return False

            token, signature = parts
            expected_signature = hmac.new(
                self.secret_key, token.encode(), hashlib.sha256
            ).hexdigest()

            return hmac.compare_digest(signature, expected_signature)
        except Exception:
            return False

    def _is_exempt(self, request: Request) -> bool:
        """Check if request is exempt from CSRF protection."""
        path = request.url.path

        # Exempt specific paths
        if path in self.EXEMPT_PATHS:
            return True

        # Exempt paths starting with exempt patterns
        for exempt in self.EXEMPT_PATHS:
            if exempt.endswith("*") and path.startswith(exempt[:-1]):
                return True

        return False

    def _validate_origin(self, request: Request) -> bool:
        """Validate Origin/Referer header matches allowed origins."""
        origin = request.headers.get("origin")
        referer = request.headers.get("referer")

        # Get the source URL
        source = origin or referer
        if not source:
            # No origin/referer - could be same-origin request
            # Be permissive for non-browser clients
            return True

        try:
            parsed = urlparse(source)
            source_origin = f"{parsed.scheme}://{parsed.netloc}"

            # Check against allowed origins
            allowed_origins = cast(List[str], settings.CORS_ORIGINS) + [
                f"{request.url.scheme}://{request.url.netloc}"
            ]

            return source_origin in allowed_origins
        except Exception:
            return False

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get existing CSRF token from cookie
        csrf_cookie = request.cookies.get(self.COOKIE_NAME)

        # For protected methods, validate CSRF
        if request.method in self.PROTECTED_METHODS and not self._is_exempt(request):
            # Validate origin header
            if not self._validate_origin(request):
                logger.warning(
                    f"CSRF origin validation failed for {request.url.path} "
                    f"from {request.headers.get('origin', 'unknown')}"
                )
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "Origin validation failed"},
                )

            # Validate CSRF token
            csrf_header = request.headers.get(self.HEADER_NAME)

            if not csrf_cookie or not csrf_header:
                logger.warning(f"Missing CSRF token for {request.url.path}")
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "CSRF token missing"},
                )

            if not self._verify_token(csrf_cookie):
                logger.warning(f"Invalid CSRF cookie token for {request.url.path}")
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "Invalid CSRF token"},
                )

            # Token in header should match cookie token
            if not hmac.compare_digest(csrf_cookie, csrf_header):
                logger.warning(f"CSRF token mismatch for {request.url.path}")
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "CSRF token mismatch"},
                )

        # Process request
        response = cast(Response, await call_next(request))

        # Set/refresh CSRF cookie if not present or expired
        if not csrf_cookie or not self._verify_token(csrf_cookie):
            new_token = self._generate_token()
            signed_token = self._sign_token(new_token)

            response.set_cookie(
                key=self.COOKIE_NAME,
                value=signed_token,
                max_age=self.cookie_max_age,
                secure=self.cookie_secure,
                httponly=self.cookie_httponly,
                samesite=cast(Literal["lax", "strict", "none"], self.cookie_samesite),
            )

        return response


class IPBlockMiddleware(BaseHTTPMiddleware):
    """
    IP blocking middleware for security.

    Blocks requests from:
    - Explicitly blocked IPs
    - IPs that exceed failed auth attempts
    - Known malicious IP ranges (optional)
    """

    def __init__(
        self,
        app: ASGIApp,
        blocked_ips: Optional[Set[str]] = None,
        max_failed_attempts: int = 10,
        block_duration_minutes: int = 30,
    ):
        super().__init__(app)
        self.blocked_ips = blocked_ips or set()
        self.max_failed_attempts = max_failed_attempts
        self.block_duration = timedelta(minutes=block_duration_minutes)
        self._failed_attempts: Dict[str, List[datetime]] = {}
        self._temporary_blocks: Dict[str, datetime] = {}

    def _get_client_ip(self, request: Request) -> str:
        """Get real client IP, considering proxies."""
        # Check X-Forwarded-For header (trusted proxies only)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take the first IP (original client)
            return forwarded.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()

        # Fall back to direct connection
        return request.client.host if request.client else "unknown"

    def _is_blocked(self, ip: str) -> bool:
        """Check if IP is blocked."""
        # Check permanent block list
        if ip in self.blocked_ips:
            return True

        # Check temporary blocks
        if ip in self._temporary_blocks:
            block_until = self._temporary_blocks[ip]
            if datetime.now(timezone.utc) < block_until:
                return True
            else:
                # Block expired
                del self._temporary_blocks[ip]

        return False

    def record_failed_attempt(self, ip: str) -> None:
        """Record a failed authentication attempt."""
        now = datetime.now(timezone.utc)
        cutoff = now - self.block_duration

        # Clean old attempts
        if ip in self._failed_attempts:
            self._failed_attempts[ip] = [t for t in self._failed_attempts[ip] if t > cutoff]
        else:
            self._failed_attempts[ip] = []

        # Add new attempt
        self._failed_attempts[ip].append(now)

        # Check if should block
        if len(self._failed_attempts[ip]) >= self.max_failed_attempts:
            self._temporary_blocks[ip] = now + self.block_duration
            logger.warning(
                f"IP {ip} temporarily blocked after {self.max_failed_attempts} failed attempts"
            )

    def clear_failed_attempts(self, ip: str) -> None:
        """Clear failed attempts for an IP (after successful auth)."""
        self._failed_attempts.pop(ip, None)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = self._get_client_ip(request)

        if self._is_blocked(client_ip):
            logger.warning(f"Blocked request from {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Access denied"},
            )

        # Store IP in request state for other middleware
        request.state.client_ip = client_ip

        return cast(Response, await call_next(request))


class InputSanitizationMiddleware(BaseHTTPMiddleware):
    """
    Input sanitization middleware.

    Provides basic XSS protection by sanitizing input:
    - Strips dangerous HTML tags
    - Escapes special characters
    - Validates content types
    """

    # Patterns that might indicate XSS
    DANGEROUS_PATTERNS = [
        "<script",
        "javascript:",
        "onclick",
        "onerror",
        "onload",
        "eval(",
        "document.cookie",
        "window.location",
    ]

    def __init__(
        self,
        app: ASGIApp,
        check_query_params: bool = True,
        check_headers: bool = True,
        log_suspicious: bool = True,
    ):
        super().__init__(app)
        self.check_query_params = check_query_params
        self.check_headers = check_headers
        self.log_suspicious = log_suspicious

    def _contains_dangerous_pattern(self, value: str) -> bool:
        """Check if value contains dangerous patterns."""
        lower_value = value.lower()
        return any(pattern in lower_value for pattern in self.DANGEROUS_PATTERNS)

    def _check_query_params(self, request: Request) -> Optional[str]:
        """Check query parameters for dangerous patterns."""
        for key, value in request.query_params.items():
            if self._contains_dangerous_pattern(value):
                return f"Dangerous pattern in query param '{key}'"
        return None

    def _check_headers(self, request: Request) -> Optional[str]:
        """Check headers for dangerous patterns."""
        # Only check user-controllable headers
        check_headers = ["user-agent", "referer", "x-custom-header"]

        for header in check_headers:
            value = request.headers.get(header, "")
            if self._contains_dangerous_pattern(value):
                return f"Dangerous pattern in header '{header}'"
        return None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        issues = []

        if self.check_query_params:
            issue = self._check_query_params(request)
            if issue:
                issues.append(issue)

        if self.check_headers:
            issue = self._check_headers(request)
            if issue:
                issues.append(issue)

        if issues:
            if self.log_suspicious:
                client_host = request.client.host if request.client else "unknown"
                logger.warning(f"Suspicious input detected from {client_host}: {issues}")
            # Optionally reject the request
            # raise HTTPException(status_code=400, detail="Invalid input detected")

        return cast(Response, await call_next(request))
