import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from unittest.mock import patch, MagicMock
import os
from fastapi import Request

@pytest.fixture(autouse=True)
def set_testing_env():
    with patch.dict(os.environ, {"TESTING": "true"}):
        yield

def test_lifespan_and_startup():
    # Using TestClient as a context manager triggers lifespan events
    with patch('src.ml.preload_critical_modules') as mock_ml_preload:
        with patch('src.pricing.preload_classical_pricers') as mock_pricing_preload:
            with TestClient(app) as client:
                mock_ml_preload.assert_called_once()
                mock_pricing_preload.assert_called_once()
                response = client.get("/health")
                assert response.status_code == 200

def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "BS-Opt API is running"}

def test_admin_only_forbidden():
    # Mock verify_token to return a payload without admin role
    async def mock_verify_token():
        return {"sub": "user123", "realm_access": {"roles": ["user"]}}
    
    app.dependency_overrides[app.extra.get("verify_token", None) or "src.auth.security.verify_token"] = mock_verify_token
    # Actually, main.py imports verify_token, so we need to override it correctly
    from src.auth.security import verify_token
    app.dependency_overrides[verify_token] = mock_verify_token

    try:
        with TestClient(app) as client:
            response = client.get("/admin-only")
            assert response.status_code == 403
    finally:
        app.dependency_overrides.clear()

def test_admin_only_success():
    # Mock verify_token to return a payload WITH admin role
    async def mock_verify_token():
        return {"sub": "admin123", "realm_access": {"roles": ["admin"]}}
    
    from src.auth.security import verify_token
    app.dependency_overrides[verify_token] = mock_verify_token

    try:
        with TestClient(app) as client:
            response = client.get("/admin-only")
            assert response.status_code == 200
            assert response.json() == {"message": "Welcome, Admin"}
    finally:
        app.dependency_overrides.clear()

def test_http_exception_handler():
    # To ensure our custom handler is used, we can trigger it manually via a route
    @app.get("/trigger-http-exception")
    async def trigger_http_exception():
        from fastapi import HTTPException
        raise HTTPException(status_code=418, detail="I am a teapot")

    with TestClient(app) as client:
        response = client.get("/trigger-http-exception")
        assert response.status_code == 418
        assert response.json() == {
            "error": "http_error",
            "message": "I am a teapot",
        }

def test_base_api_exception_handler():
    from src.api.exceptions import BaseAPIException
    
    @app.get("/trigger-base-exception")
    async def trigger_base_exception():
        raise BaseAPIException(
            status_code=400,
            error="test_error",
            message="Test message",
            details={"foo": "bar"}
        )
    
    with TestClient(app) as client:
        response = client.get("/trigger-base-exception")
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "test_error"
        assert data["message"] == "Test message"
        assert data["details"] == {"foo": "bar"}

def test_instrument_requests_middleware():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200

def test_get_context_testing():
    from src.api.main import get_context
    mock_request = MagicMock(spec=Request)
    with patch.dict(os.environ, {"TESTING": "true"}):
        ctx = get_context(mock_request)
        assert ctx == {}

def test_get_context_production():
    from src.api.main import get_context
    mock_request = MagicMock(spec=Request)
    with patch.dict(os.environ, {"TESTING": "false"}):
        ctx = get_context(mock_request)
        assert ctx == {"request": mock_request}

def test_prometheus_dir_creation(mocker):
    # Mock Counter and Histogram to avoid duplicate registration errors on reload
    with patch('prometheus_client.Counter'):
        with patch('prometheus_client.Histogram'):
            with patch('os.path.exists', return_value=False):
                with patch('os.makedirs') as mock_makedirs:
                    import src.api.main
                    importlib.reload(src.api.main)
                    mock_makedirs.assert_called_with(ANY, exist_ok=True)

import importlib
from unittest.mock import ANY

def test_lifespan_shutdown_flush():
    mock_producer = MagicMock()
    app.state.audit_producer = mock_producer
    
    with patch('src.ml.preload_critical_modules'):
        with patch('src.pricing.preload_classical_pricers'):
            with TestClient(app) as client:
                pass # Startup and shutdown logic runs
    
    mock_producer.flush.assert_called_once_with(timeout=5)
    app.state.audit_producer = None # Clean up