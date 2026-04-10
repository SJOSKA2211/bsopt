import pytest
import json
import uuid
from unittest.mock import MagicMock, patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from api.middleware.logging import RequestLoggingMiddleware, StructuredLogger


def test_structured_logger_full():
    sl = StructuredLogger("test_logger")
    sl.set_default_fields(app="bsopt")

    with (
        patch("api.middleware.logging.logging.Logger.debug") as mock_debug,
        patch("api.middleware.logging.logging.Logger.info") as mock_info,
        patch("api.middleware.logging.logging.Logger.warning") as mock_warn,
        patch("api.middleware.logging.logging.Logger.error") as mock_error,
        patch("api.middleware.logging.logging.Logger.critical") as mock_crit,
    ):
        sl.debug("dbg")
        sl.info("inf")
        sl.warning("wrn")
        sl.error("err")
        sl.critical("crit")
        sl.exception("exc")

        assert mock_debug.called
        assert mock_info.called
        assert mock_warn.called
        assert mock_error.call_count == 2
        assert mock_crit.called


def test_request_logging_internals():
    middleware = RequestLoggingMiddleware(MagicMock())

    # Redact headers
    headers = {"Authorization": "secret", "Content-Type": "json"}
    redacted = middleware._redact_headers(headers)
    assert redacted["Authorization"] == "[REDACTED]"
    assert redacted["Content-Type"] == "json"

    # Redact params
    params = {"token": "secret", "q": "search"}
    redacted_p = middleware._redact_params(params)
    assert redacted_p["token"] == "[REDACTED]"
    assert redacted_p["q"] == "search"

    # Truncate body
    middleware.max_body_length = 5
    body = "1234567890"
    truncated = middleware._truncate_body(body)
    assert truncated.startswith("12345")
    assert "[truncated" in truncated


def test_request_logging_middleware_basic():
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware, persist_to_db=False)

    @app.get("/test")
    async def test_route():
        return {"ok": True}

    client = TestClient(app)
    with patch("api.middleware.logging.request_logger.info") as mock_log:
        client.get("/test")
        assert mock_log.called


def test_request_logging_malformed_json():
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware, log_request_body=True, persist_to_db=False)

    @app.post("/malformed")
    async def malformed(request: Request):
        return {"ok": True}

    client = TestClient(app)
    with patch("api.middleware.logging.request_logger.info") as mock_log:
        # Use valid UTF-8 but not JSON
        response = client.post("/malformed", content=b"not a json")
        assert response.status_code == 200
        log_entry = json.loads(mock_log.call_args[0][0])
        assert log_entry["body"] == "not a json"


def test_request_logging_user_info():
    app = FastAPI()

    class MockUser:
        def __init__(self):
            self.id = uuid.uuid4()
            self.email = "user@test.com"
            self.tier = "pro"

    @app.middleware("http")
    async def add_user(request: Request, call_next):
        request.state.user = MockUser()
        return await call_next(request)

    app.add_middleware(RequestLoggingMiddleware, persist_to_db=False)

    @app.get("/user-test")
    async def user_test():
        return {"ok": True}

    client = TestClient(app)

    with patch("api.middleware.logging.request_logger.info") as mock_log:
        client.get("/user-test")
        log_entry = json.loads(mock_log.call_args[0][0])
        # The middleware now masks emails by default
        assert "***" in log_entry["user_email"]



@pytest.mark.asyncio
@patch("src.database.SessionLocal")
async def test_persist_log_full(mock_session_local):
    with patch("src.database.models.RequestLog"):
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        middleware = RequestLoggingMiddleware(MagicMock(), persist_to_db=True)
        log_entry = {
            "request_id": "test",
            "method": "GET",
            "path": "/test",
            "status_code": 200,
            "duration_ms": 10,
            "client_ip": "127.0.0.1",
            "user_id": str(uuid.uuid4())
        }
        
        await middleware._persist_log(log_entry, MagicMock())
        
        mock_session.add.assert_called()
        mock_session.commit.assert_called()

        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware, persist_to_db=True)

        @app.get("/persist")
        async def persist(request: Request):
            # Make sure request.state.user is NOT a MagicMock when logging tries to access it
            class RealUser:
                id = uuid.uuid4()
                email = "p@test.com"
                tier = "free"

            request.state.user = RealUser()
            return {"ok": True}

        client = TestClient(app)
        client.get("/persist")

        # Give background task time to run
        import time
        time.sleep(0.2)

        mock_session.add.assert_called()
        mock_session.commit.assert_called()
        mock_session.close.assert_called()


def test_request_logging_error_capture():
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware, persist_to_db=False)

    @app.get("/error")
    async def error_route():
        raise ValueError("test error")

    client = TestClient(app, raise_server_exceptions=False)

    with patch("api.middleware.logging.request_logger.info") as mock_log:
        response = client.get("/error")
        assert response.status_code == 500
        log_entry = json.loads(mock_log.call_args[0][0])
        assert log_entry["status_code"] == 500
        assert "error" in log_entry
