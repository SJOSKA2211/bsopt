import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import Request, HTTPException
from src.security.rate_limit import RateLimiter, rate_limit

@pytest.mark.asyncio
async def test_rate_limit_default():
    # Test backward compatibility instance
    limiter = rate_limit

    # redis_client.pipeline() is synchronous, so mock_redis should be MagicMock
    mock_redis = MagicMock()
    mock_pipeline = MagicMock()
    mock_redis.pipeline.return_value = mock_pipeline
    mock_pipeline.execute = AsyncMock(return_value=(1, True)) # Count = 1
    mock_redis.ttl = AsyncMock(return_value=30)

    request = MagicMock(spec=Request)
    request.state.user = None
    request.client.host = "127.0.0.1"
    request.state.rate_limit_remaining = None

    await limiter(request, mock_redis)

    # Verify pipeline calls
    assert mock_redis.pipeline.called
    assert mock_pipeline.incr.called
    assert mock_pipeline.expire.called
    assert mock_pipeline.execute.called

    # Verify request state updated
    assert request.state.rate_limit_remaining is not None

@pytest.mark.asyncio
async def test_rate_limit_exceeded():
    limiter = RateLimiter(requests=5, window=60)

    mock_redis = MagicMock()
    mock_pipeline = MagicMock()
    mock_redis.pipeline.return_value = mock_pipeline
    mock_pipeline.execute = AsyncMock(return_value=(6, True)) # Count = 6 (Limit is 5)
    mock_redis.ttl = AsyncMock(return_value=30)

    request = MagicMock(spec=Request)
    request.state.user = None
    request.client.host = "127.0.0.1"
    request.url.path = "/login"

    with pytest.raises(HTTPException) as excinfo:
        await limiter(request, mock_redis)

    assert excinfo.value.status_code == 429
    assert excinfo.value.headers["X-RateLimit-Limit"] == "5"
    assert excinfo.value.headers["X-RateLimit-Remaining"] == "0"

@pytest.mark.asyncio
async def test_rate_limit_custom():
    limiter = RateLimiter(requests=10, window=60)

    mock_redis = MagicMock()
    mock_pipeline = MagicMock()
    mock_redis.pipeline.return_value = mock_pipeline
    mock_pipeline.execute = AsyncMock(return_value=(1, True))
    mock_redis.ttl = AsyncMock(return_value=30)

    request = MagicMock(spec=Request)
    request.state.user = None
    request.client.host = "127.0.0.1"
    request.url.path = "/register"

    await limiter(request, mock_redis)

    # Check key format
    # The key should include the path
    # key = f"rate_limit:{identifier}:{request.url.path}:{int(time.time()) // window}"
    args, _ = mock_pipeline.incr.call_args
    assert "/register" in args[0]

@pytest.mark.asyncio
async def test_rate_limit_tier_logic():
    # Test that default logic uses tier
    limiter = RateLimiter() # No explicit limit

    mock_redis = MagicMock()
    mock_pipeline = MagicMock()
    mock_redis.pipeline.return_value = mock_pipeline
    mock_pipeline.execute = AsyncMock(return_value=(1, True))
    mock_redis.ttl = AsyncMock(return_value=30)

    request = MagicMock(spec=Request)
    user = MagicMock()
    user.id = "user123"
    user.tier = "pro" # Limit should be 1000
    request.state.user = user
    request.client.host = "127.0.0.1"

    await limiter(request, mock_redis)

    assert request.state.rate_limit_limit == 1000 # PRO tier limit
