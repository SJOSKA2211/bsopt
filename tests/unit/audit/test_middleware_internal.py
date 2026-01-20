import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import Request, Response
from src.audit.middleware import AuditMiddleware
import json

@pytest.mark.asyncio
async def test_audit_middleware_with_id_user():
    app = MagicMock()
    middleware = AuditMiddleware(app)
    
    mock_request = MagicMock(spec=Request)
    mock_request.state.user = MagicMock()
    mock_request.state.user.id = "u123"
    mock_request.url.path = "/test"
    mock_request.method = "GET"
    mock_request.client = MagicMock()
    mock_request.client.host = "1.2.3.4"
    mock_request.headers = {"user-agent": "test-agent"}
    mock_request.app.state.audit_producer = None
    
    mock_response = MagicMock(spec=Response)
    mock_response.status_code = 200
    
    async def call_next(req):
        return mock_response
        
    response = await middleware.dispatch(mock_request, call_next)
    assert response == mock_response

@pytest.mark.asyncio
async def test_audit_middleware_with_anonymous_user():
    app = MagicMock()
    middleware = AuditMiddleware(app)
    
    mock_request = MagicMock(spec=Request)
    mock_request.state.user = object() # Not a dict, has no id
    mock_request.url.path = "/test"
    mock_request.method = "GET"
    mock_request.client = None
    mock_request.headers = {}
    mock_request.app.state.audit_producer = None
    
    async def call_next(req):
        return MagicMock(spec=Response, status_code=200)
        
    await middleware.dispatch(mock_request, call_next)

@pytest.mark.asyncio
async def test_audit_middleware_with_producer_success():
    mock_producer = MagicMock()
    app = MagicMock()
    middleware = AuditMiddleware(app, producer=mock_producer)
    
    mock_request = MagicMock(spec=Request)
    mock_request.state.user = {"sub": "u1"}
    mock_request.url.path = "/test"
    mock_request.method = "GET"
    mock_request.client = None
    mock_request.headers = {}
    
    mock_response = MagicMock(spec=Response, status_code=200)
    async def call_next(req):
        return mock_response
        
    await middleware.dispatch(mock_request, call_next)
    mock_producer.produce.assert_called_once()
    mock_producer.poll.assert_called_once_with(0)

@pytest.mark.asyncio
async def test_audit_middleware_with_producer_exception():
    mock_producer = MagicMock()
    mock_producer.produce.side_effect = Exception("Kafka down")
    app = MagicMock()
    middleware = AuditMiddleware(app, producer=mock_producer)
    
    mock_request = MagicMock(spec=Request)
    mock_request.state.user = {"sub": "u1"}
    mock_request.url.path = "/test"
    mock_request.method = "GET"
    mock_request.client = None
    mock_request.headers = {}
    
    async def call_next(req):
        return MagicMock(spec=Response, status_code=200)
        
    # Should not raise exception
    await middleware.dispatch(mock_request, call_next)
    mock_producer.produce.assert_called_once()
