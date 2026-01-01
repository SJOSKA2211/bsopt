import time
import logging
from enum import Enum
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class CircuitBreaker:
    """
    Implementation of the Circuit Breaker pattern to increase fault tolerance.
    """
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED

    def __call__(self, func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit Breaker: State changed to HALF_OPEN")
                else:
                    raise Exception("Circuit Breaker is OPEN. Request rejected.")

            try:
                result = func(*args, **kwargs)
                
                if self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info("Circuit Breaker: State changed to CLOSED")
                
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.error(f"Circuit Breaker: State changed to OPEN due to {self.failure_count} failures. Last error: {e}")
                
                raise e
        return wrapper

# Global instances for different services
pricing_circuit = CircuitBreaker(failure_threshold=10, recovery_timeout=60)
db_circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=30)
