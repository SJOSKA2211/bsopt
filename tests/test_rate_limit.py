import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import Request, HTTPException
from src.api.dependencies.rate_limit import LoginRateLimiter

@pytest.mark.asyncio
async def test_rate_limiter_allows_under_limit():
    limiter = LoginRateLimiter(max_attempts=2, window_seconds=60)

    mock_redis = AsyncMock()
    # eval returns attempts
    mock_redis.eval.return_value = 1

    request = MagicMock(spec=Request)
    request.client.host = "127.0.0.1"

    with pytest.MonkeyPatch().context() as m:
        m.setattr("src.api.dependencies.rate_limit.get_redis", lambda: mock_redis)

        await limiter(request)

        mock_redis.eval.assert_called_once()
        args = mock_redis.eval.call_args
        # redis.eval(script, numkeys, key, arg)
        assert args[0][2] == "login_limit:127.0.0.1"

@pytest.mark.asyncio
async def test_rate_limiter_blocks_over_limit():
    limiter = LoginRateLimiter(max_attempts=2, window_seconds=60)

    mock_redis = AsyncMock()
    mock_redis.eval.return_value = 3 # Over limit

    request = MagicMock(spec=Request)
    request.client.host = "127.0.0.1"

    with pytest.MonkeyPatch().context() as m:
        m.setattr("src.api.dependencies.rate_limit.get_redis", lambda: mock_redis)

        with pytest.raises(HTTPException) as excinfo:
            await limiter(request)

        assert excinfo.value.status_code == 429
        assert "Too many login attempts" in excinfo.value.detail

@pytest.mark.asyncio
async def test_rate_limiter_handles_redis_failure():
    limiter = LoginRateLimiter(max_attempts=2)

    mock_redis = AsyncMock()
    mock_redis.eval.side_effect = Exception("Redis down")

    request = MagicMock(spec=Request)
    request.client.host = "127.0.0.1"

    with pytest.MonkeyPatch().context() as m:
        m.setattr("src.api.dependencies.rate_limit.get_redis", lambda: mock_redis)

        # Should not raise exception (fail open)
        await limiter(request)

@pytest.mark.asyncio
async def test_rate_limiter_ignores_forwarded_ip():
    """Ensure rate limiter ignores X-Forwarded-For and uses client.host"""
    limiter = LoginRateLimiter(max_attempts=2)

    mock_redis = AsyncMock()
    mock_redis.eval.return_value = 1

    request = MagicMock(spec=Request)
    request.client.host = "127.0.0.1"
    # Even if X-Forwarded-For is present
    request.headers.get.return_value = "10.0.0.1"

    with pytest.MonkeyPatch().context() as m:
        m.setattr("src.api.dependencies.rate_limit.get_redis", lambda: mock_redis)

        await limiter(request)

        # Should use 127.0.0.1, not 10.0.0.1
        args = mock_redis.eval.call_args
        assert args[0][2] == "login_limit:127.0.0.1"
