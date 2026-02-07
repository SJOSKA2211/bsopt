from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog

from src.shared import observability


@pytest.fixture
def mock_gateway():
    with patch("src.shared.observability.push_to_gateway") as mock:
        yield mock

@pytest.fixture
def mock_httpx():
    with patch("src.shared.observability.httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value
        instance.post = AsyncMock()
        yield instance

def test_off_heap_processor():
    event_dict = {"event": "test", "high_frequency": True}
    with pytest.raises(structlog.DropEvent):
        observability._off_heap_processor(None, None, event_dict)

def test_tune_gc():
    observability.tune_gc("high_frequency")
    observability.tune_gc("analytical")
    # Just ensure it doesn't crash; gc settings are global

def test_push_metrics(mock_gateway):
    observability.push_metrics("test_job")
    # It runs in a thread pool, so we can't easily assert call immediately without wait
    # But code should execute

@pytest.mark.asyncio
async def test_post_grafana_annotation(mock_httpx):
    mock_httpx.post.return_value.status_code = 200
    with patch.dict(observability.os.environ, {"GRAFANA_URL": "http://grafana"}):
        result = await observability.post_grafana_annotation("test")
        assert result is True

@pytest.mark.asyncio
async def test_logging_middleware():
    request = MagicMock()
    request.headers.get.return_value = "req-id"
    request.client.host = "1.2.3.4"
    request.url.path = "/test"
    request.method = "GET"
    
    async def call_next(req):
        resp = MagicMock()
        req.state.request_id = "req-id" # Middleware sets this
        resp.status_code = 200
        resp.headers = {}
        return resp
    
    response = await observability.logging_middleware(request, call_next)
    assert response.headers["X-Request-ID"] == "req-id"
