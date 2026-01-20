import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import Request, Response, HTTPException
from src.api.middleware.security import (
    SecurityHeadersMiddleware, 
    CSRFMiddleware, 
    IPBlockMiddleware, 
    InputSanitizationMiddleware
)
import starlette.status as status

@pytest.mark.asyncio
async def test_security_headers_middleware():
    app = MagicMock()
    middleware = SecurityHeadersMiddleware(app)
    request = MagicMock(spec=Request)
    request.url.path = "/test"
    request.url.scheme = "https"
    
    async def call_next(req):
        return Response(content="ok")
        
    response = await middleware.dispatch(request, call_next)
    assert "Strict-Transport-Security" in response.headers
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" in response.headers

@pytest.mark.asyncio
async def test_csrf_middleware_exempt():
    app = MagicMock()
    # Mock settings
    with patch("src.api.middleware.security.settings") as mock_settings:
        mock_settings.JWT_SECRET = "secret"
        mock_settings.is_production = False
        middleware = CSRFMiddleware(app)
        request = MagicMock(spec=Request)
        request.method = "POST"
        request.url.path = "/api/v1/auth/login"
        request.cookies = {}
        
        async def call_next(req):
            return Response(content="ok")
            
        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_ip_block_middleware():
    app = MagicMock()
    middleware = IPBlockMiddleware(app, blocked_ips={"1.2.3.4"})
    
    # Blocked IP
    request = MagicMock(spec=Request)
    request.client.host = "1.2.3.4"
    request.headers = {}
    
    async def call_next(req):
        return Response(content="ok")
        
    with pytest.raises(HTTPException) as exc:
        await middleware.dispatch(request, call_next)
    assert exc.value.status_code == 403

@pytest.mark.asyncio
async def test_input_sanitization_middleware():
    app = MagicMock()
    middleware = InputSanitizationMiddleware(app)
    
    request = MagicMock(spec=Request)
    request.query_params = {"q": "<script>alert(1)</script>"}
    request.headers = {}
    request.client.host = "127.0.0.1"
    
    async def call_next(req):
        return Response(content="ok")
        
    # Should log warning but continue (as configured in current implementation)
    with patch("src.api.middleware.security.logger") as mock_logger:
        await middleware.dispatch(request, call_next)
        assert mock_logger.warning.called
