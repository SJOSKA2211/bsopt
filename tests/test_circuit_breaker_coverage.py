import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from src.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    DistributedCircuitBreaker,
    pricing_circuit,
)


@pytest.mark.asyncio
async def test_circuit_breaker_state_transitions():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
    assert cb.state == CircuitState.CLOSED

    # We need to wrap it ourselves because __call__ returns a decorator
    def test_func():
        return "ok"

    def fail_func():
        raise Exception("fail")

    wrapped_fail = cb(fail_func)
    wrapped_ok = cb(test_func)

    # First failure
    with pytest.raises(Exception, match="fail"):
        wrapped_fail()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 1

    # Second failure -> Open
    with pytest.raises(Exception, match="fail"):
        wrapped_fail()
    assert cb.state == CircuitState.OPEN

    # Call while open
    with pytest.raises(Exception, match="Circuit Breaker is OPEN"):
        wrapped_ok()

    # Wait for recovery timeout
    await asyncio.sleep(0.15)
    # Still open until next call

    # Call -> Half-Open -> Success -> Closed
    assert wrapped_ok() == "ok"
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


@pytest.mark.asyncio
async def test_distributed_circuit_breaker():
    mock_redis = AsyncMock()
    cb = DistributedCircuitBreaker(
        name="dist", redis_client=mock_redis, failure_threshold=2, recovery_timeout=60
    )
    mock_redis.get.return_value = None

    async def test_func():
        return "ok"

    wrapped = cb(test_func)
    assert await wrapped() == "ok"
    mock_redis.get.side_effect = [b"OPEN", b"1000000"]
    assert await wrapped() == "ok"


@pytest.mark.asyncio
async def test_distributed_circuit_breaker_still_open():
    mock_redis = AsyncMock()
    cb = DistributedCircuitBreaker(
        name="dist", redis_client=mock_redis, failure_threshold=2, recovery_timeout=60
    )
    # Mock state as OPEN and last_failure as recent
    mock_redis.get.side_effect = [b"OPEN", str(int(time.time())).encode()]

    async def test_func():
        return "ok"

    wrapped = cb(test_func)
    with pytest.raises(Exception, match="is OPEN"):
        await wrapped()


@pytest.mark.asyncio
async def test_distributed_circuit_breaker_sync_func():
    mock_redis = AsyncMock()
    cb = DistributedCircuitBreaker(
        name="dist", redis_client=mock_redis, failure_threshold=2, recovery_timeout=60
    )
    mock_redis.get.return_value = None

    def sync_func():
        return "ok"

    wrapped = cb(sync_func)
    assert await wrapped() == "ok"


@pytest.mark.asyncio
async def test_distributed_circuit_breaker_fail_below_threshold():
    mock_redis = AsyncMock()
    cb = DistributedCircuitBreaker(
        name="dist", redis_client=mock_redis, failure_threshold=5, recovery_timeout=60
    )
    mock_redis.get.return_value = None  # CLOSED
    mock_redis.incr.return_value = 1  # below threshold

    async def fail_func():
        raise Exception("fail")

    wrapped = cb(fail_func)
    with pytest.raises(Exception, match="fail"):
        await wrapped()
    # Should not set state to OPEN
    assert not any(
        call.args[0] == CircuitState.OPEN.value
        for call in mock_redis.set.call_args_list
        if call.args
    )


@pytest.mark.asyncio
async def test_distributed_circuit_breaker_fail():
    mock_redis = AsyncMock()
    cb = DistributedCircuitBreaker(
        name="dist", redis_client=mock_redis, failure_threshold=2, recovery_timeout=60
    )
    mock_redis.get.return_value = None  # CLOSED
    mock_redis.incr.return_value = 2  # hits threshold

    async def fail_func():
        raise Exception("fail")

    wrapped = cb(fail_func)
    with pytest.raises(Exception, match="fail"):
        await wrapped()
    assert mock_redis.set.called


def test_pricing_circuit_global():

    # Use the global instance

    # Reset it first since it's global

    pricing_circuit.state = CircuitState.CLOSED

    pricing_circuit.failure_count = 0

    def fail():
        raise Exception("fail")

    wrapped = pricing_circuit(fail)

    # pricing_circuit has threshold 10

    for _ in range(10):

        with pytest.raises(Exception, match="fail"):

            wrapped()

    with pytest.raises(Exception, match="Circuit Breaker is OPEN"):

        wrapped()

    @pytest.mark.asyncio
    async def test_distributed_circuit_breaker_helpers():

        mock_redis = AsyncMock()

        cb = DistributedCircuitBreaker(name="test", redis_client=mock_redis)

        # _set_state

        await cb._set_state(CircuitState.OPEN, expiry=10)

        mock_redis.set.assert_called_with("test:cb_state", "OPEN", ex=10)

        # _get_failures (none)

        mock_redis.get.return_value = None

        assert await cb._get_failures() == 0

        # _get_failures (exists)

        mock_redis.get.return_value = b"5"

        assert await cb._get_failures() == 5

        # _get_last_failure_time (none)

        mock_redis.get.return_value = None

        assert await cb._get_last_failure_time() == 0

        # _get_last_failure_time (exists)

        mock_redis.get.return_value = b"123456"

        assert await cb._get_last_failure_time() == 123456

        # _set_last_failure_time

        await cb._set_last_failure_time(999, expiry=5)

        mock_redis.set.assert_called_with("test:cb_last_failure", 999, ex=5)
