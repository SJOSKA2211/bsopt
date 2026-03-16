import json
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api.middleware.logging import RequestLoggingMiddleware

app = FastAPI()


@app.get("/test")
async def route_test():
    return {"message": "success"}


@app.get("/health")
async def route_health():
    return {"status": "ok"}


@app.get("/error")
async def route_error():
    raise ValueError("Test Error")


app.add_middleware(RequestLoggingMiddleware, persist_to_db=False)

client = TestClient(app)


def test_request_logging_basic():
    with patch("services.api.middleware.logging.request_logger.log") as mock_log:
        response = client.get("/test")
        assert response.status_code == 200
        mock_log.assert_called()
        # Verify JSON content of log
        args, kwargs = mock_log.call_args
        log_json = json.loads(args[1])
        assert log_json["path"] == "/test"
        assert log_json["method"] == "GET"


def test_request_logging_redaction():
    with patch("services.api.middleware.logging.request_logger.log") as mock_log:
        client.get(
            "/test",
            params={"password": "secret_pass"},
            headers={"Authorization": "Bearer token"},
        )
        args, kwargs = mock_log.call_args
        log_json = json.loads(args[1])
        assert log_json["query_params"]["password"] == "[REDACTED]"
        assert log_json["headers"]["authorization"] == "[REDACTED]"


def test_request_logging_skip():
    with patch("services.api.middleware.logging.request_logger.log") as mock_log:
        client.get("/health")
        mock_log.assert_not_called()


def test_request_logging_error():
    with patch("services.api.middleware.logging.request_logger.log") as mock_log:
        with pytest.raises(ValueError):
            client.get("/error")

        # Finally block still runs
        mock_log.assert_called()
        args, kwargs = mock_log.call_args
        log_json = json.loads(args[1])
        assert log_json["status_code"] == 500
        assert log_json["error"]["type"] == "ValueError"
