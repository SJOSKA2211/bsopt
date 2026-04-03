import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.shared.utils.circuit_breaker import (
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
    async def test_func():
        return "ok"

    async def fail_func():
        raise Exception("fail")

    wrapped_fail = cb(fail_func)
    wrapped_ok = cb(test_func)

    # First failure
    with pytest.raises(Exception, match="fail"):
        await wrapped_fail()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 1

    # Second failure -> Open
    with pytest.raises(Exception, match="fail"):
        await wrapped_fail()
    assert cb.state == CircuitState.OPEN

    # Call while open
    with pytest.raises(Exception, match="is OPEN"):
        await wrapped_ok()

    # Wait for recovery timeout
    await asyncio.sleep(0.15)
    # Still open until next call

    # Call -> Half-Open -> Success -> Closed
    assert await wrapped_ok() == "ok"
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


@pytest.mark.asyncio
async def test_distributed_circuit_breaker():
    mock_redis = AsyncMock()

    # register_script returns an object that is CALLABLE and returns a COROUTINE
    mock_script = MagicMock()
    mock_script.return_value = AsyncMock(return_value=b"CLOSED")()
    mock_redis.register_script.return_value = mock_script

    cb = DistributedCircuitBreaker(
        name="dist", redis_client=mock_redis, failure_threshold=2, recovery_timeout=60
    )

    async def test_func():
        return "ok"

    wrapped = cb(test_func)
    assert await wrapped() == "ok"

    # Test transitioning to HALF_OPEN
    mock_script.return_value = AsyncMock(return_value=b"HALF_OPEN")()
    assert await wrapped() == "ok"
    mock_redis.delete.assert_called()


@pytest.mark.asyncio
async def test_distributed_circuit_breaker_still_open():
    mock_redis = AsyncMock()
    mock_script = MagicMock()
    mock_script.return_value = AsyncMock(return_value=b"OPEN")()
    mock_redis.register_script.return_value = mock_script

    cb = DistributedCircuitBreaker(
        name="dist", redis_client=mock_redis, failure_threshold=2, recovery_timeout=60
    )

    async def test_func():
        return "ok"

    wrapped = cb(test_func)
    with pytest.raises(Exception, match="is OPEN"):
        await wrapped()


@pytest.mark.asyncio
async def test_distributed_circuit_breaker_sync_func():
    mock_redis = AsyncMock()
    mock_script = MagicMock()
    mock_script.return_value = AsyncMock(return_value=b"CLOSED")()
    mock_redis.register_script.return_value = mock_script

    cb = DistributedCircuitBreaker(
        name="dist", redis_client=mock_redis, failure_threshold=2, recovery_timeout=60
    )

    def sync_func():
        return "ok"

    wrapped = cb(sync_func)
    assert await wrapped() == "ok"


@pytest.mark.asyncio
async def test_distributed_circuit_breaker_fail_below_threshold():
    mock_redis = AsyncMock()
    mock_script = MagicMock()
    mock_script.return_value = AsyncMock(return_value=b"CLOSED")()
    mock_redis.register_script.return_value = mock_script
    mock_redis.incr.return_value = 1  # below threshold
    mock_redis.get.return_value = b"1"

    async def fail_func():
        raise Exception("fail")

    cb = DistributedCircuitBreaker(
        name="dist", redis_client=mock_redis, failure_threshold=5, recovery_timeout=60
    )
    wrapped = cb(fail_func)
    with pytest.raises(Exception, match="fail"):
        await wrapped()

    # Verify set was NOT called with OPEN
    open_calls = [call for call in mock_redis.set.call_args_list if "OPEN" in str(call)]
    assert len(open_calls) == 0


@pytest.mark.asyncio
async def test_distributed_circuit_breaker_fail():
    mock_redis = AsyncMock()
    mock_script = MagicMock()
    mock_script.return_value = AsyncMock(return_value=b"CLOSED")()
    mock_redis.register_script.return_value = mock_script
    mock_redis.incr.return_value = 2  # hits threshold
    mock_redis.get.return_value = b"2"

    async def fail_func():
        raise Exception("fail")

    cb = DistributedCircuitBreaker(
        name="dist", redis_client=mock_redis, failure_threshold=2, recovery_timeout=60
    )
    wrapped = cb(fail_func)
    with pytest.raises(Exception, match="fail"):
        await wrapped()
    # verify set OPEN called
    mock_redis.set.assert_any_call("dist:cb_state", "OPEN", ex=60)


@pytest.mark.asyncio
async def test_pricing_circuit_global():
    # Reset global state for test
    from src.shared.utils.circuit_breaker import CircuitState

    pricing_circuit.state = CircuitState.CLOSED
    pricing_circuit.failure_count = 0

    async def fail():
        raise Exception("fail")

    wrapped = pricing_circuit(fail)

    for _ in range(10):
        with pytest.raises(Exception, match="fail"):
            await wrapped()

    with pytest.raises(Exception, match="is OPEN"):
        await wrapped()


@pytest.mark.asyncio
async def test_distributed_circuit_breaker_helpers():
    # Tested via __call__ in cases above.
    pass
