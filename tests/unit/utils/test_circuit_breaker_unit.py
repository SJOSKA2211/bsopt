import pytest
import time
import asyncio
from src.utils.circuit_breaker import InMemoryCircuitBreaker, DistributedCircuitBreaker, CircuitState

def test_in_memory_circuit_breaker():
    cb = InMemoryCircuitBreaker(failure_threshold=2, recovery_timeout=1)
    
    @cb
    def risky_func(fail=False):
        if fail: raise ValueError("Fail")
        return "success"
    
    # Success
    assert risky_func() == "success"
    
    # Fail 1
    with pytest.raises(ValueError):
        risky_func(fail=True)
    assert cb.state == CircuitState.CLOSED
    
    # Fail 2 -> OPEN
    with pytest.raises(ValueError):
        risky_func(fail=True)
    assert cb.state == CircuitState.OPEN
    
    # Should reject when OPEN
    with pytest.raises(Exception, match="Circuit Breaker is OPEN"):
        risky_func()
        
    # Wait for recovery
    time.sleep(1.1)
    # Should move to HALF_OPEN and then CLOSED on success
    assert risky_func() == "success"
    assert cb.state == CircuitState.CLOSED

@pytest.mark.asyncio
async def test_distributed_circuit_breaker(mocker):
    mock_redis = mocker.Mock()
    # threshold=2, recovery=1
    cb = DistributedCircuitBreaker("test", mock_redis, failure_threshold=2, recovery_timeout=1)
    
    # Mock redis methods
    mock_redis.get = mocker.AsyncMock(return_value=None) # Default CLOSED
    mock_redis.set = mocker.AsyncMock()
    mock_redis.incr = mocker.AsyncMock(return_value=1)
    mock_redis.delete = mocker.AsyncMock()
    
    @cb
    async def async_func(fail=False):
        if fail: raise ValueError("Fail")
        return "success"
    
    # 1. Successful call
    assert await async_func() == "success"
    
    # 2. Failure 1
    with pytest.raises(ValueError):
        await async_func(fail=True)
    
    # 3. Failure 2 -> OPEN
    mock_redis.incr.return_value = 2
    with pytest.raises(ValueError):
        await async_func(fail=True)
    mock_redis.set.assert_any_call("test:cb_state", "OPEN", ex=1)
    
    # 4. Request when OPEN (last_failure_time is very recent)
    now = int(time.time())
    # First call to get() returns state, second returns last_failure_time
    mock_redis.get.side_effect = [b"OPEN", str(now).encode()]
    with pytest.raises(Exception, match="is OPEN"):
        await async_func()
        
    # 5. Recovery to HALF_OPEN -> CLOSED
    # last_failure_time is in the past
    mock_redis.get.side_effect = [b"OPEN", b"1000"] 
    assert await async_func() == "success"
    
    # Check that it set state to HALF_OPEN then CLOSED
    mock_redis.set.assert_any_call("test:cb_state", "HALF_OPEN", ex=None)
    mock_redis.set.assert_any_call("test:cb_state", "CLOSED", ex=None)

@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_distributed_cb_internal_getters(mocker):
    mock_redis = mocker.Mock()
    cb = DistributedCircuitBreaker("test", mock_redis)
    
    # Correctly mock as AsyncMock
    mock_redis.get = mocker.AsyncMock()
    
    # test _get_failures
    mock_redis.get.side_effect = [b"5", None]
    assert await cb._get_failures() == 5
    assert await cb._get_failures() == 0
    
    # test _get_last_failure_time
    mock_redis.get.side_effect = [b"12345", None]
    assert await cb._get_last_failure_time() == 12345
    assert await cb._get_last_failure_time() == 0

@pytest.mark.asyncio
async def test_distributed_cb_sync_func(mocker):
    mock_redis = mocker.Mock()
    cb = DistributedCircuitBreaker("test", mock_redis)
    mock_redis.get = mocker.AsyncMock(return_value=None)
    
    @cb
    def sync_func():
        return "sync_success"
        
    assert await sync_func() == "sync_success"