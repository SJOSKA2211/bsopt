import time
import logging
from enum import Enum
from functools import wraps
from typing import Callable, Any, Optional, cast
import asyncio
import inspect

import redis.asyncio as redis
from src.config import settings

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class InMemoryCircuitBreaker:
    """
    In-memory implementation of the Circuit Breaker pattern.
    Suitable for single-process applications or testing.
    """
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED

    def reset(self):
        """Reset the circuit breaker to CLOSED state."""
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED

    def __call__(self, func: Callable):
        sig = inspect.signature(func)
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if self.state == CircuitState.OPEN:
                    if time.time() - self.last_failure_time > self.recovery_timeout:
                        self.state = CircuitState.HALF_OPEN
                        logger.info("Circuit Breaker: State changed to HALF_OPEN")
                    else:
                        raise Exception("Circuit Breaker is OPEN. Request rejected.")

                try:
                    result = await func(*args, **kwargs)
                    if self.state == CircuitState.HALF_OPEN:
                        self.state = CircuitState.CLOSED
                        self.failure_count = 0
                        logger.info("Circuit Breaker: State changed to CLOSED")
                    return result
                except Exception as e:
                    self._handle_failure(e)
                    raise e
            async_wrapper.__signature__ = sig  # type: ignore
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
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
                    self._handle_failure(e)
                    raise e
            sync_wrapper.__signature__ = sig  # type: ignore
            return sync_wrapper

    def _handle_failure(self, e: Exception):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error("circuit_breaker_state_changed_to_open", name="InMemoryCircuitBreaker", failures=self.failure_count, error=str(e), exc_info=True)


# Alias for backward compatibility
CircuitBreaker = InMemoryCircuitBreaker


import re # Import regex module

# ... (previous code) ...

class DistributedCircuitBreaker:
    """
    Distributed Circuit Breaker using Redis for state management.
    Suitable for multi-process/multi-instance applications.
    """
    REDIS_KEY_STATE = "{name}:cb_state"
    REDIS_KEY_FAILURES = "{name}:cb_failures"
    REDIS_KEY_LAST_FAILURE = "{name}:cb_last_failure"

    def _sanitize_name(self, name: str) -> str:
        """Sanitizes the name to prevent Redis Key Injection."""
        # Allow alphanumeric, hyphens, and underscores. Replace others with underscore.
        return re.sub(r'[^\w-]', '_', name)

    def __init__(self, name: str, redis_client: redis.Redis, failure_threshold: int = 5, recovery_timeout: int = 30):
        # --- SECURITY: Sanitize name to prevent Redis Key Injection ---
        self.name = self._sanitize_name(name)
        self.redis_client = redis_client
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout


    async def _get_state(self) -> CircuitState:
        state_str = await self.redis_client.get(self.REDIS_KEY_STATE.format(name=self.name))
        return CircuitState(state_str.decode() if state_str else CircuitState.CLOSED.value)

    async def _set_state(self, state: CircuitState, expiry: Optional[int] = None) -> None:
        await self.redis_client.set(self.REDIS_KEY_STATE.format(name=self.name), state.value, ex=expiry)

    async def _get_failures(self) -> int:
        failures = await self.redis_client.get(self.REDIS_KEY_FAILURES.format(name=self.name))
        return int(failures) if failures else 0

    async def _increment_failures(self) -> int:
        return cast(int, await self.redis_client.incr(self.REDIS_KEY_FAILURES.format(name=self.name)))

    async def _reset_failures(self) -> None:
        await self.redis_client.delete(self.REDIS_KEY_FAILURES.format(name=self.name))

    async def _get_last_failure_time(self) -> int:
        last_failure = await self.redis_client.get(self.REDIS_KEY_LAST_FAILURE.format(name=self.name))
        return int(last_failure) if last_failure else 0

    async def _set_last_failure_time(self, timestamp: int, expiry: Optional[int] = None) -> None:
        await self.redis_client.set(self.REDIS_KEY_LAST_FAILURE.format(name=self.name), timestamp, ex=expiry)

    def __call__(self, func: Callable):
        sig = inspect.signature(func)
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_state = await self._get_state()
            current_time = int(time.time())

            if current_state == CircuitState.OPEN:
                last_failure = await self._get_last_failure_time()
                if current_time - last_failure > self.recovery_timeout:
                    await self._set_state(CircuitState.HALF_OPEN)
                    logger.info("circuit_breaker_state_changed_to_half_open", name=self.name)
                    current_state = CircuitState.HALF_OPEN
                else:
                    raise Exception(f"Circuit Breaker '{self.name}' is OPEN. Request rejected.")

            try:
                # Execute the original function (it must be awaitable if it's async)
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs) # type: ignore
                
                if current_state == CircuitState.HALF_OPEN:
                    await self._set_state(CircuitState.CLOSED)
                    await self._reset_failures()
                    logger.info("circuit_breaker_state_changed_to_closed", name=self.name)
                
                return result
            except Exception as e:
                failures = await self._increment_failures()
                await self._set_last_failure_time(current_time)
                
                if failures >= self.failure_threshold:
                    await self._set_state(CircuitState.OPEN, expiry=self.recovery_timeout) # Open for recovery_timeout
                    logger.error("circuit_breaker_state_changed_to_open", name=self.name, failures=failures, error=str(e), exc_info=True)
                
                raise e
        wrapper.__signature__ = sig # type: ignore
        return wrapper

# Global instances for different services (now InMemory by default if not specified)
# Users would typically explicitly initialize DistributedCircuitBreaker where needed.
pricing_circuit = InMemoryCircuitBreaker(failure_threshold=10, recovery_timeout=60)
db_circuit = InMemoryCircuitBreaker(failure_threshold=5, recovery_timeout=30)