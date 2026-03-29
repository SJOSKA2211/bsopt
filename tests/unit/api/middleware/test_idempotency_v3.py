from unittest.mock import AsyncMock, MagicMock

import msgspec
import pytest
from fastapi import Request, Response

from src.api.middleware.idempotency import IdempotencyMiddleware

@pytest.fixture
def mock_redis():
    return AsyncMock()

@pytest.fixture
def mock_app():
    return MagicMock()

@pytest.mark.asyncio
async def test_idempotency_skip_get(mock_app, mock_redis):
    middleware = IdempotencyMiddleware(mock_app, mock_redis)
    request = MagicMock(spec=Request)
    request.method = "GET"
    request.headers = {}

    call_next = AsyncMock(return_value=Response(status_code=200))
    res = await middleware.dispatch(request, call_next)

    assert res.status_code == 200
    assert not mock_redis.get.called

@pytest.mark.asyncio
async def test_idempotency_cache_hit(mock_app, mock_redis):
    middleware = IdempotencyMiddleware(mock_app, mock_redis)
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.url.path = "/test"
    request.headers = {"X-Idempotency-Key": "key123"}

    # Mock cached data
    cached_data = {
        "status_code": 201,
        "content": '{"ok": true}',
        "headers": {"content-type": "application/json"},
    }
    mock_redis.get.return_value = msgspec.json.encode(cached_data)

    call_next = AsyncMock()
    res = await middleware.dispatch(request, call_next)

    assert res.status_code == 201
    assert res.headers["X-Idempotency-Cache"] == "HIT"
    assert not call_next.called

@pytest.mark.asyncio
async def test_idempotency_lock_conflict(mock_app, mock_redis):
    middleware = IdempotencyMiddleware(mock_app, mock_redis)
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.url.path = "/test"
    request.headers = {"X-Idempotency-Key": "key123"}

    mock_redis.get.return_value = None
    mock_redis.set.return_value = False  # Lock acquisition failed

    call_next = AsyncMock()
    res = await middleware.dispatch(request, call_next)

    assert res.status_code == 409
    assert "already in progress" in str(res.body)

@pytest.mark.asyncio
async def test_idempotency_success_caching(mock_app, mock_redis):
    middleware = IdempotencyMiddleware(mock_app, mock_redis)
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.url.path = "/test"
    request.headers = {"X-Idempotency-Key": "key123"}

    mock_redis.get.return_value = None
    mock_redis.set.return_value = True  # Lock acquired

    mock_response = Response(content='{"new": true}', status_code=200)
    call_next = AsyncMock(return_value=mock_response)

    res = await middleware.dispatch(request, call_next)

    assert res.status_code == 200
    assert mock_redis.set.call_count == 2  # 1 for lock, 1 for cache
    assert mock_redis.delete.called  # Lock released
