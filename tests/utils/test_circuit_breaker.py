import time
from unittest.mock import AsyncMock

import pytest

from src.utils.circuit_breaker import (
    CircuitState,
    DistributedCircuitBreaker,
    InMemoryCircuitBreaker,
)


def test_local_circuit_breaker_flow():
    cb = InMemoryCircuitBreaker(failure_threshold=2, recovery_timeout=1)

    def failing_func():
        raise ValueError("Fail")

    def success_func():
        return "Success"

    # First failure
    with pytest.raises(ValueError):
        cb(failing_func)()
    assert cb.state == CircuitState.CLOSED

    # Second failure -> OPEN
    with pytest.raises(ValueError):
        cb(failing_func)()
    assert cb.state == CircuitState.OPEN

    # Immediate call while OPEN -> Exception
    with pytest.raises(Exception, match="Circuit Breaker is OPEN"):
        cb(success_func)()

    # Wait for recovery
    time.sleep(1.1)

    # Call while OPEN after timeout -> HALF_OPEN -> Success -> CLOSED
    assert cb(success_func)() == "Success"
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_redis_circuit_breaker_flow():
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    mock_redis.incr.return_value = 1

    cb = DistributedCircuitBreaker(
        name="test", redis_client=mock_redis, failure_threshold=2, recovery_timeout=1
    )

    async def async_failing_func():
        raise ValueError("Fail")

    async def async_success_func():
        return "Success"

    # Mock first failure
    mock_redis.get.return_value = None  # State CLOSED
    with pytest.raises(ValueError):
        await cb(async_failing_func)()

    # Mock second failure -> OPEN
    mock_redis.incr.return_value = 2
    with pytest.raises(ValueError):
        await cb(async_failing_func)()

    # Mock OPEN state
    mock_redis.get.side_effect = [
        b"OPEN",
        b"1000000000",
    ]  # State OPEN, Last failure long ago
    # Current time is > 1000000000 + 1

    # Mock transition to HALF_OPEN
    res = await cb(async_success_func)()
    assert res == "Success"
    mock_redis.set.assert_any_call("test:cb_state", "CLOSED", ex=None)


@pytest.mark.asyncio
async def test_redis_circuit_breaker_open_rejection():
    mock_redis = AsyncMock()
    mock_redis.get.side_effect = [b"OPEN", b"2000000000"]  # OPEN and not timed out

    cb = DistributedCircuitBreaker(
        name="test", redis_client=mock_redis, recovery_timeout=100
    )

    async def some_func():
        return "ok"

    with pytest.raises(Exception, match="is OPEN. Request rejected"):
        await cb(some_func)()


@pytest.mark.asyncio
async def test_redis_circuit_breaker_sync_func():
    mock_redis = AsyncMock()
    mock_redis.get.return_value = b"CLOSED"

    cb = DistributedCircuitBreaker(name="test", redis_client=mock_redis)

    def sync_func():
        return "sync_ok"

    assert await cb(sync_func)() == "sync_ok"


@pytest.mark.asyncio
async def test_redis_circuit_breaker_get_failures():
    mock_redis = AsyncMock()
    mock_redis.get.return_value = b"5"

    cb = DistributedCircuitBreaker(name="test", redis_client=mock_redis)
    assert await cb._get_failures() == 5

    mock_redis.get.return_value = None
    assert await cb._get_failures() == 0
