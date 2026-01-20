import pytest
import json
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import Request, Response
from src.api.middleware.logging import RequestLoggingMiddleware, StructuredLogger
from src.database.models import User
import uuid

@pytest.mark.asyncio
async def test_redact_headers():
    middleware = RequestLoggingMiddleware(None)
    headers = {
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
        "X-API-Key": "my-key"
    }
    redacted = middleware._redact_headers(headers)
    assert redacted["Authorization"] == "[REDACTED]"
    assert redacted["X-API-Key"] == "[REDACTED]"
    assert redacted["Content-Type"] == "application/json"

@pytest.mark.asyncio
async def test_redact_params():
    middleware = RequestLoggingMiddleware(None)
    params = {"token": "123", "symbol": "AAPL"}
    redacted = middleware._redact_params(params)
    assert redacted["token"] == "[REDACTED]"
    assert redacted["symbol"] == "AAPL"

@pytest.mark.asyncio
async def test_truncate_body():
    middleware = RequestLoggingMiddleware(None, max_body_length=10)
    body = "a" * 20
    truncated = middleware._truncate_body(body)
    assert "truncated" in truncated
    # Check that it actually truncated the original body part
    assert body[:10] in truncated

@pytest.mark.asyncio
async def test_get_user_info():
    middleware = RequestLoggingMiddleware(None)
    request = MagicMock(spec=Request)
    
    class State:
        pass
    request.state = State()
    
    # Mock user object with real values to avoid MagicMock serialization issues
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.email = "test@example.com"
    user.tier = "pro"
    request.state.user = user
    
    info = middleware._get_user_info(request)
    assert info["user_id"] == str(user.id)
    assert info["user_email"] == "test@example.com"

@pytest.mark.asyncio
async def test_logging_middleware_dispatch():
    app = MagicMock()
    middleware = RequestLoggingMiddleware(app, log_request_body=True, persist_to_db=False)
    
    request = MagicMock(spec=Request)
    request.url.path = "/api/v1/test"
    request.method = "POST"
    request.headers = {"user-agent": "test"}
    request.query_params = {}
    
    class State:
        pass
    request.state = State()
    request.state.request_id = "test-id"
    request.state.client_ip = "127.0.0.1"
    
    # Set user info to avoid MagicMocks in log_entry
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.email = "test@example.com"
    user.tier = "free"
    request.state.user = user
    
    request.body = AsyncMock(return_value=b'{"password": "123"}')
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    
    async def call_next(req):
        return Response(content="ok", status_code=200)
        
    with patch("src.api.middleware.logging.request_logger") as mock_log:
        await middleware.dispatch(request, call_next)
        assert mock_log.log.called
        log_call = mock_log.log.call_args
        log_data = json.loads(log_call[0][1])
        assert log_data["request_id"] == "test-id"
        assert "[REDACTED]" in log_data["body"]

def test_structured_logger():
    s_logger = StructuredLogger("test")
    s_logger.set_default_fields(app="bsopt")
    
    with patch.object(s_logger.logger, "info") as mock_info:
        s_logger.info("message", extra="field")
        assert mock_info.called
        log_data = json.loads(mock_info.call_args[0][0])
        assert log_data["message"] == "message"
        assert log_data["app"] == "bsopt"
        assert log_data["extra"] == "field"