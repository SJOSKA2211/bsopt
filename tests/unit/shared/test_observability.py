import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import structlog
import os
import time
from prometheus_client import Summary, Counter, Gauge, Histogram, push_to_gateway, REGISTRY
from fastapi import Request, Response
import httpx
from datetime import datetime, timezone
import json

from src.shared.observability import (
    setup_logging,
    logging_middleware,
    push_metrics,
    post_grafana_annotation,
    SCRAPE_DURATION,
    SCRAPE_ERRORS,
    TRAINING_DURATION,
    MODEL_ACCURACY,
    MODEL_RMSE,
    DATA_DRIFT_SCORE,
    KS_TEST_SCORE,
    PERFORMANCE_DRIFT_ALERT,
    TRAINING_ERRORS,
    HESTON_FELLER_MARGIN,
    CALIBRATION_DURATION,
    HESTON_R_SQUARED,
    HESTON_PARAMS_FRESHNESS
)

@pytest.fixture(autouse=True)
def mock_env_vars():
    with patch.dict(os.environ, {
        "PUSHGATEWAY_URL": "http://mock-pushgateway:9091",
        "GRAFANA_URL": "http://mock-grafana:3000"
    }, clear=True):
        yield

@pytest.fixture
def mock_structlog_logger():
    with patch('structlog.get_logger') as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        yield mock_logger

@pytest.fixture
def mock_httpx_client():
    with patch('httpx.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": "Success"}
        mock_post.return_value = mock_response
        yield mock_post, mock_response

# --- Tests for setup_logging ---
def test_setup_logging():
    with patch('structlog.configure') as mock_configure, \
         patch('structlog.PrintLoggerFactory') as mock_print_logger_factory:
        setup_logging()
        mock_configure.assert_called_once()
        args, kwargs = mock_configure.call_args
        processors = kwargs['processors']
        assert any(isinstance(p, structlog.processors.JSONRenderer) for p in processors)
        assert kwargs['logger_factory'] is mock_print_logger_factory.return_value

# --- Tests for logging_middleware ---
@pytest.mark.asyncio
async def test_logging_middleware_success(mock_structlog_logger):
    mock_request = MagicMock(spec=Request)
    mock_request.method = "GET"
    mock_request.url.path = "/test"
    mock_request.client.host = "192.168.1.100"

    mock_response = MagicMock(spec=Response)
    mock_response.status_code = 200

    call_next_mock = AsyncMock(return_value=mock_response)

    response = await logging_middleware(mock_request, call_next_mock)

    call_next_mock.assert_awaited_once_with(mock_request)
    mock_structlog_logger.info.assert_called_once()
    
    logged_kwargs = mock_structlog_logger.info.call_args.kwargs
    assert logged_kwargs["method"] == "GET"
    assert logged_kwargs["path"] == "/test"
    assert logged_kwargs["status_code"] == 200
    assert "duration_ms" in logged_kwargs
    assert logged_kwargs["client_ip"] == "192.168.1.xxx" # Masked IP
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_logging_middleware_no_client_ip(mock_structlog_logger):
    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.url.path = "/data"
    mock_request.client = None # No client IP

    mock_response = MagicMock(spec=Response)
    mock_response.status_code = 201

    call_next_mock = AsyncMock(return_value=mock_response)

    await logging_middleware(mock_request, call_next_mock)

    logged_kwargs = mock_structlog_logger.info.call_args.kwargs
    assert logged_kwargs["client_ip"] == "unknown"

# --- Tests for push_metrics ---
def test_push_metrics_success(mock_structlog_logger):
    with patch('src.shared.observability.push_to_gateway') as mock_push_to_gateway:
        push_metrics("test_job")
        mock_push_to_gateway.assert_called_once_with(
            "http://mock-pushgateway:9091",
            job="test_job",
            registry=REGISTRY
        )
        mock_structlog_logger.info.assert_called_with("metrics_pushed", job="test_job", gateway="http://mock-pushgateway:9091")

def test_push_metrics_no_url(mock_structlog_logger):
    with patch.dict(os.environ, {}, clear=True): # Clear PUSHGATEWAY_URL
        push_metrics("test_job")
        mock_structlog_logger.debug.assert_called_with("metrics_push_skipped", reason="no_gateway_url")

def test_push_metrics_failure(mock_structlog_logger):
    with patch('src.shared.observability.push_to_gateway', side_effect=Exception("Push error")) as mock_push_to_gateway:
        push_metrics("test_job")
        mock_push_to_gateway.assert_called_once()
        mock_structlog_logger.error.assert_called_with("metrics_push_failed", error="Push error")

# --- Tests for post_grafana_annotation ---
def test_post_grafana_annotation_success(mock_httpx_client, mock_structlog_logger):
    mock_post, mock_response = mock_httpx_client
    message = "Test annotation"
    tags = ["tag1", "tag2"]
    
    result = post_grafana_annotation(message, tags)
    
    assert result is True
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "http://mock-grafana:3000/api/annotations"
    assert kwargs['json']['text'] == message
    assert kwargs['json']['tags'] == tags
    assert "time" in kwargs['json']
    mock_structlog_logger.info.assert_called_once()

def test_post_grafana_annotation_no_url(mock_httpx_client, mock_structlog_logger):
    mock_post, _ = mock_httpx_client
    with patch.dict(os.environ, {}, clear=True): # Clear GRAFANA_URL
        result = post_grafana_annotation("Test message")
        assert result is False
        mock_post.assert_not_called()
        mock_structlog_logger.debug.assert_called_with("grafana_annotation_skipped", reason="GRAFANA_URL not set")

def test_post_grafana_annotation_http_error(mock_httpx_client, mock_structlog_logger):
    mock_post, mock_response = mock_httpx_client
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError("Bad Request", request=MagicMock(), response=mock_response)
    mock_response.text = "Error details" # Simulate response text for debugging
    
    result = post_grafana_annotation("Test message")
    
    assert result is False
    mock_post.assert_called_once()
    mock_structlog_logger.error.assert_called_once()
    logged_kwargs = mock_structlog_logger.error.call_args.kwargs
    assert logged_kwargs["reason"] == "HTTPStatusError"
    assert "Bad Request" in logged_kwargs["error"]
    assert logged_kwargs["response_text"] == "Error details"

def test_post_grafana_annotation_request_error(mock_httpx_client, mock_structlog_logger):
    mock_post, _ = mock_httpx_client
    mock_post.side_effect = httpx.RequestError("Network error", request=MagicMock())
    
    result = post_grafana_annotation("Test message")
    
    assert result is False
    mock_post.assert_called_once()
    mock_structlog_logger.error.assert_called_once()
    logged_kwargs = mock_structlog_logger.error.call_args.kwargs
    assert logged_kwargs["reason"] == "RequestError"
    assert "Network error" in logged_kwargs["error"]

def test_post_grafana_annotation_unknown_error(mock_httpx_client, mock_structlog_logger):
    mock_post, _ = mock_httpx_client
    mock_post.side_effect = Exception("Unknown error")
    
    result = post_grafana_annotation("Test message")
    
    assert result is False
    mock_post.assert_called_once()
    mock_structlog_logger.error.assert_called_once()
    logged_kwargs = mock_structlog_logger.error.call_args.kwargs
    assert logged_kwargs["reason"] == "UnknownError"
    assert "Unknown error" in logged_kwargs["error"]