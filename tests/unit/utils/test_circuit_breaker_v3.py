import asyncio

import pytest

from src.utils.circuit_breaker import CircuitState, InMemoryCircuitBreaker


@pytest.mark.asyncio
async def test_in_memory_circuit_breaker_success():
    cb = InMemoryCircuitBreaker(failure_threshold=2, recovery_timeout=1)
    
    @cb
    async def success():
        return "ok"
        
    res = await success()
    assert res == "ok"
    assert cb.state == CircuitState.CLOSED

@pytest.mark.asyncio
async def test_in_memory_circuit_breaker_failure_and_open():
    cb = InMemoryCircuitBreaker(failure_threshold=2, recovery_timeout=1)
    
    @cb
    async def fail():
        raise ValueError("boom")
        
    # First failure
    with pytest.raises(ValueError):
        await fail()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 1
    
    # Second failure -> opens
    with pytest.raises(ValueError):
        await fail()
    assert cb.state == CircuitState.OPEN
    
    # Third call -> fails fast with general Exception
    with pytest.raises(Exception) as excinfo:
        await fail()
    assert "is OPEN" in str(excinfo.value)

@pytest.mark.asyncio
async def test_in_memory_circuit_breaker_half_open():
    cb = InMemoryCircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
    
    @cb
    async def task():
        if cb.state == CircuitState.HALF_OPEN:
            return "recovered"
        raise ValueError("boom")
    
    # Open it
    with pytest.raises(ValueError):
        await task()
    assert cb.state == CircuitState.OPEN
    
    # Wait for recovery
    await asyncio.sleep(0.2)
    
    # Half-open -> Success -> Closed
    res = await task()
    assert res == "recovered"
    assert cb.state == CircuitState.CLOSED
