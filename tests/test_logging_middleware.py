from unittest.mock import MagicMock, patch

import pytest
from fastapi import Request

from api.middleware.logging import RequestLoggingMiddleware


@pytest.mark.asyncio
async def test_persist_log_correctly_uses_session_local():
    """
    Verifies that _persist_log uses SessionLocal directly instead of get_session generator.
    This ensures that database persistence works correctly and doesn't crash due to
    AttributeError on generator object.
    """
    app = MagicMock()
    middleware = RequestLoggingMiddleware(app)

    log_entry = {
        "request_id": "123",
        "method": "GET",
        "path": "/test",
        "status_code": 200,
        "duration_ms": 100,
        "client_ip": "127.0.0.1",
        "user_id": "00000000-0000-0000-0000-000000000000",
        "query_params": {"q": "test"},
        "headers": {"header": "value"},
    }
    request = MagicMock(spec=Request)

    # Mock SessionLocal to return a MockSession
    mock_session = MagicMock()

    # We patch SessionLocal because the fix changes import from get_session to SessionLocal
    with patch("src.database.SessionLocal", return_value=mock_session) as mock_session_local:
        with patch("api.middleware.logging.logger") as mock_logger:
            with patch("src.database.models.RequestLog"):
                await middleware._persist_log(log_entry, request)

            # Verify that SessionLocal was instantiated
            mock_session_local.assert_called_once()

            # Verify that add and commit were called on the session
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()

            # Verify no error logged (which would happen if exception occurred)
            assert not mock_logger.error.called