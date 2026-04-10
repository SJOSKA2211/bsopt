from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request, Response

from api.middleware.security import SecurityHeadersMiddleware


@pytest.mark.asyncio
async def test_security_headers_middleware():
    app = AsyncMock()
    middleware = SecurityHeadersMiddleware(app)

    # Mock a secure request to trigger HSTS
    request = MagicMock(spec=Request)
    request.url.path = "/api/v1/pricing"
    request.url.scheme = "https"
    request.scope = {"type": "http", "scheme": "https"}

    async def call_next(req):
        return Response(content="ok", status_code=200)

    response = await middleware.dispatch(request, call_next)

    headers = {k.lower(): v for k, v in response.headers.items()}
    assert headers["x-frame-options"] == "DENY"
    assert headers["x-content-type-options"] == "nosniff"
    # HSTS should be there for https
    assert "strict-transport-security" in headers
    assert "content-security-policy" in headers
    assert headers["referrer-policy"] == "strict-origin-when-cross-origin"


@pytest.mark.asyncio
async def test_security_no_cache_headers():
    app = AsyncMock()
    middleware = SecurityHeadersMiddleware(app)

    # Path in NO_CACHE_PATTERNS
    request = MagicMock(spec=Request)
    request.url.path = "/api/v1/auth/login"
    request.scope = {"type": "http", "scheme": "http"}

    async def call_next(req):
        return Response(content="ok", status_code=200)

    response = await middleware.dispatch(request, call_next)
    headers = {k.lower(): v for k, v in response.headers.items()}
    assert "no-store" in headers["cache-control"]


@pytest.mark.asyncio
async def test_security_middleware_custom_csp():
    app = AsyncMock()
    custom_csp = {"default-src": ["'none'"]}
    middleware = SecurityHeadersMiddleware(app, csp_directives=custom_csp)

    request = MagicMock(spec=Request)
    request.url.path = "/test"
    request.scope = {"type": "http", "scheme": "http"}

    async def call_next(req):
        return Response(content="ok", status_code=200)

    response = await middleware.dispatch(request, call_next)
    headers = {k.lower(): v for k, v in response.headers.items()}
    assert "default-src 'none'" in headers["content-security-policy"]
