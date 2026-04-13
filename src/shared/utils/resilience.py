"""
Resilience Utilities for BS-Opt (Manifold)
==
Implements Circuit Breaker, Exponential Backoff with Jitter,
and advanced error handling for external dependencies.
"""

import asyncio
import random
import time
from collections.abc import Callable
from enum import Enum
from functools import wraps
from typing import TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[type[Exception], ...] = (Exception,),
):
    """
    Decorator for retrying a function with exponential backoff and optional jitter.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            for i in range(retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if i == retries - 1:
                        raise

                    actual_delay = delay
                    if jitter:
                        actual_delay += random.uniform(0, delay * 0.25)

                    logger.warning(
                        "retry_attempt_failed",
                        func=func.__name__,
                        attempt=i + 1,
                        retries=retries,
                        delay=f"{actual_delay:.2f}s",
                        error=str(e),
                    )
                    await asyncio.sleep(actual_delay)
                    delay *= backoff_factor

        return wrapper

    return decorator


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Robust Circuit Breaker implementation with configurable thresholds.
    Focuses on zero-local-state (using class attributes or being passed as singleton).
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exceptions: tuple = (Exception,),
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions

        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0.0

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    logger.info("circuit_breaker_half_open", name=self.name)
                    self.state = CircuitState.HALF_OPEN
                else:
                    logger.warning("circuit_breaker_restricted", name=self.name)
                    raise RuntimeError(f"Circuit {self.name} is OPEN")

            try:
                result = await func(*args, **kwargs)

                # If we were half-open or open and succeeded, reset
                if self.state != CircuitState.CLOSED:
                    logger.info("circuit_breaker_reset", name=self.name)
                    self.reset()
                return result

            except self.expected_exceptions as e:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    logger.error("circuit_breaker_tripped", name=self.name, error=str(e))
                    self.state = CircuitState.OPEN

                raise e

        return wrapper

    def reset(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0.0


# Pre-defined breakers
yfinance_breaker = CircuitBreaker("yfinance", failure_threshold=10, recovery_timeout=60)
nse_breaker = CircuitBreaker("nse", failure_threshold=5, recovery_timeout=30)
db_breaker = CircuitBreaker("database", failure_threshold=20, recovery_timeout=10)
