import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI, Request, Response
from src.audit.middleware import AuditMiddleware
import json

@pytest.mark.asyncio
async def test_audit_middleware_calls_kafka():
    app = FastAPI()
    
    # Mock Producer
    mock_producer = MagicMock()
    
    app.add_middleware(AuditMiddleware, producer=mock_producer, topic="test-audit")
    
    @app.get("/test")
    async def test_route(request: Request):
        request.state.user = {"sub": "user123"}
        return {"message": "ok"}
    
    from fastapi.testclient import TestClient
    client = TestClient(app)
    
    # Execute request
    response = client.get("/test")
    assert response.status_code == 200
    
    # Verify Producer.produce was called
    assert mock_producer.produce.called
    args, kwargs = mock_producer.produce.call_args
    assert args[0] == "test-audit"
    
    # Verify payload
    payload = json.loads(args[1])
    assert payload["method"] == "GET"
    assert payload["path"] == "/test"
    assert payload["status_code"] == 200
    assert payload["user_id"] == "user123"
    assert "timestamp" in payload

@pytest.mark.asyncio
async def test_audit_middleware_exception_handling():
    app = FastAPI()
    mock_producer = MagicMock()
    # Simulate exception during produce
    mock_producer.produce.side_effect = Exception("Kafka Error")
    
    app.add_middleware(AuditMiddleware, producer=mock_producer, topic="test-audit")
    
    @app.get("/test")
    async def test_route(request: Request):
        return {"message": "ok"}
    
    from fastapi.testclient import TestClient
    client = TestClient(app)
    
    # Request should still succeed even if audit fails
    response = client.get("/test")
    assert response.status_code == 200
    assert mock_producer.produce.called