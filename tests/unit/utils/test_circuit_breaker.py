import pytest
import time
from src.utils.circuit_breaker import CircuitBreaker, CircuitState

def test_circuit_breaker_transitions():
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
    
    @breaker
    def failing_func():
        raise ValueError("Fail")
    
    @breaker
    def success_func():
        return "OK"
    
    # First failure
    with pytest.raises(ValueError):
        failing_func()
    assert breaker.state == CircuitState.CLOSED
    
    # Second failure -> OPEN
    with pytest.raises(ValueError):
        failing_func()
    assert breaker.state == CircuitState.OPEN
    
    # Rejected while OPEN
    with pytest.raises(Exception, match="is OPEN"):
        success_func()
        
    # Wait for timeout -> HALF_OPEN
    time.sleep(1.1)
    assert success_func() == "OK"
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0
