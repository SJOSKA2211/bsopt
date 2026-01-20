import pytest
import time
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from src.utils.circuit_breaker import InMemoryCircuitBreaker, DistributedCircuitBreaker, CircuitState

def test_in_memory_circuit_breaker_sync():
    cb = InMemoryCircuitBreaker(failure_threshold=2, recovery_timeout=1)
    
    def fail_func():
        raise ValueError("Fail")
    
    wrapped = cb(fail_func)
    
    # 1st fail
    with pytest.raises(ValueError):
        wrapped()
    assert cb.state == CircuitState.CLOSED
    
    # 2nd fail -> OPEN
    with pytest.raises(ValueError):
        wrapped()
    assert cb.state == CircuitState.OPEN
    
    # Check OPEN rejection
    with pytest.raises(Exception, match="is OPEN"):
        wrapped()
        
    # Wait for recovery
    time.sleep(1.1)
    
    # Should move to HALF_OPEN and then CLOSED on success
    def success_func():
        return "OK"
    wrapped_success = cb(success_func)
    assert wrapped_success() == "OK"
    assert cb.state == CircuitState.CLOSED

@pytest.mark.asyncio
async def test_in_memory_circuit_breaker_async():
    cb = InMemoryCircuitBreaker(failure_threshold=1, recovery_timeout=1)
    
    async def async_fail():
        raise ValueError("Async Fail")
        
    wrapped = cb(async_fail)
    
    with pytest.raises(ValueError):
        await wrapped()
    assert cb.state == CircuitState.OPEN
    
    with pytest.raises(Exception, match="is OPEN"):
        await wrapped()

@pytest.mark.asyncio
async def test_distributed_circuit_breaker():
    mock_redis = AsyncMock()
    # Mock _get_state to return CLOSED initially
    mock_redis.get.return_value = b"CLOSED"
    
    cb = DistributedCircuitBreaker("test", mock_redis, failure_threshold=2, recovery_timeout=1)
    
    async def my_func():
        return "Data"
        
    wrapped = cb(my_func)
    assert await wrapped() == "Data"
    
    # Test failure logic
    async def fail_func():
        raise ValueError("Remote Fail")
    wrapped_fail = cb(fail_func)
    
    # Mock increment to return 2 (threshold reached)
    mock_redis.incr.return_value = 2
    
    with pytest.raises(ValueError):
        await wrapped_fail()
        
    # Verify Redis interactions
    mock_redis.set.assert_any_call("test:cb_state", "OPEN", ex=1)

def test_circuit_breaker_reset():
    cb = InMemoryCircuitBreaker()
    cb.state = CircuitState.OPEN
    cb.reset()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0
