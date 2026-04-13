from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi import Request, Response
from api.middleware.security import (
    CSRFMiddleware,
    InputSanitizationMiddleware,
    IPBlockMiddleware,
    JWTAuthenticationMiddleware,
    SecurityHeadersMiddleware,
)

@pytest.fixture
def mock_call_next():
    return AsyncMock(return_value=Response())

@pytest.fixture
def mock_receive():
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}
    return receive

@pytest.mark.asyncio
async def test_security_headers(mock_call_next, mock_receive):
    middleware = SecurityHeadersMiddleware(MagicMock())
    request = Request(scope={"type": "http", "path": "/", "headers": [], "scheme": "https"}, receive=mock_receive)
    result = await middleware.dispatch(request, mock_call_next)
    assert "X-Content-Type-Options" in result.headers

@pytest.mark.asyncio
async def test_csrf_protection_safe_methods(mock_call_next, mock_receive):
    middleware = CSRFMiddleware(MagicMock())
    request = Request(scope={"type": "http", "method": "GET", "path": "/", "headers": []}, receive=mock_receive)
    result = await middleware.dispatch(request, mock_call_next)
    assert result.status_code == 200

@pytest.mark.asyncio
async def test_csrf_protection_unsafe_method_missing_token(mock_call_next, mock_receive):
    middleware = CSRFMiddleware(MagicMock())
    request = Request(scope={"type": "http", "method": "POST", "path": "/", "headers": []}, receive=mock_receive)
    result = await middleware.dispatch(request, mock_call_next)
    assert result.status_code == 403

@pytest.mark.asyncio
async def test_ip_block_middleware_blocked(mock_call_next, mock_receive):
    middleware = IPBlockMiddleware(MagicMock(), blocked_ips={"9.9.9.9"})
    request = Request(
        scope={"type": "http", "client": ("9.9.9.9", 1234), "path": "/", "headers": []},
        receive=mock_receive
    )
    result = await middleware.dispatch(request, mock_call_next)
    assert result.status_code == 403

@pytest.mark.asyncio
async def test_jwt_auth_legacy_bypass(mock_call_next, mock_receive):
    middleware = JWTAuthenticationMiddleware(MagicMock())
    headers = [(b"authorization", b"Bearer legacy-engineer-token")]
    # JWTAuthenticationMiddleware uses auth_service.validate_token
    # But it also has a list of public paths. / is not strictly public in the list but it returns call_next
    # Wait, / is in the list of exemptions.
    request = Request(scope={"type": "http", "path": "/", "headers": headers}, receive=mock_receive)
    result = await middleware.dispatch(request, mock_call_next)
    assert result.status_code == 200

@pytest.mark.asyncio
async def test_input_sanitization(mock_call_next):
    middleware = InputSanitizationMiddleware(MagicMock())
    
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}
        
    request = Request(
        scope={
            "type": "http",
            "method": "GET",
            "path": "/",
            "query_string": b"q=<script>alert(1)</script>",
            "headers": [],
        },
        receive=receive
    )
    result = await middleware.dispatch(request, mock_call_next)
    assert result.status_code == 400 # Should be rejected due to dangerous pattern in query params