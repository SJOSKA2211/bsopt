import asyncio
import time
from collections.abc import Callable
from enum import Enum
from functools import wraps
from typing import Any, cast

import redis.asyncio as redis
import structlog

logger = structlog.get_logger(__name__)


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
        """Resets the circuit breaker state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        logger.info("circuit_breaker_reset", mechanism="in_memory")

    def __call__(self, func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("circuit_breaker_half_open", mechanism="in_memory")
                else:
                    raise Exception("Circuit Breaker is OPEN. Request rejected.")

            try:
                result = func(*args, **kwargs)

                if self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info("circuit_breaker_closed", mechanism="in_memory")

                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.error(
                        "circuit_breaker_open",
                        mechanism="in_memory",
                        failures=self.failure_count,
                        error=str(e),
                    )

                raise e

        return wrapper


# Alias for backward compatibility
CircuitBreaker = InMemoryCircuitBreaker


class DistributedCircuitBreaker:
    """
    Distributed Circuit Breaker using Redis for state management.
    Suitable for multi-process/multi-instance applications.
    """

    REDIS_KEY_STATE = "{name}:cb_state"
    REDIS_KEY_FAILURES = "{name}:cb_failures"
    REDIS_KEY_LAST_FAILURE = "{name}:cb_last_failure"

    def __init__(
        self,
        name: str,
        redis_client: redis.Redis,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
    ):
        # Security: Validate name to prevent Redis key injection
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            raise ValueError(f"Invalid circuit breaker name: {name}")

        self.name = name
        self.redis_client = redis_client
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

    async def _get_state(self) -> CircuitState:
        state_str = await self.redis_client.get(
            self.REDIS_KEY_STATE.format(name=self.name)
        )
        return CircuitState(
            state_str.decode() if state_str else CircuitState.CLOSED.value
        )

    async def _set_state(self, state: CircuitState, expiry: int | None = None) -> None:
        await self.redis_client.set(
            self.REDIS_KEY_STATE.format(name=self.name), state.value, ex=expiry
        )

    async def _get_failures(self) -> int:
        failures = await self.redis_client.get(
            self.REDIS_KEY_FAILURES.format(name=self.name)
        )
        return int(failures) if failures else 0

    async def _increment_failures(self) -> int:
        return cast(
            int,
            await self.redis_client.incr(
                self.REDIS_KEY_FAILURES.format(name=self.name)
            ),
        )

    async def _reset_failures(self) -> None:
        await self.redis_client.delete(self.REDIS_KEY_FAILURES.format(name=self.name))

    async def _get_last_failure_time(self) -> int:
        last_failure = await self.redis_client.get(
            self.REDIS_KEY_LAST_FAILURE.format(name=self.name)
        )  # pragma: no cover
        return int(last_failure) if last_failure else 0  # pragma: no cover

    async def _set_last_failure_time(
        self, timestamp: int, expiry: int | None = None
    ) -> None:
        await self.redis_client.set(
            self.REDIS_KEY_LAST_FAILURE.format(name=self.name), timestamp, ex=expiry
        )

    def __call__(self, func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_state = await self._get_state()
            current_time = int(time.time())

            if current_state == CircuitState.OPEN:
                last_failure = await self._get_last_failure_time()
                if current_time - last_failure > self.recovery_timeout:
                    await self._set_state(CircuitState.HALF_OPEN)
                    logger.info(
                        "circuit_breaker_half_open",
                        name=self.name,
                        mechanism="distributed",
                    )
                    current_state = CircuitState.HALF_OPEN
                else:
                    raise Exception(
                        f"Circuit Breaker '{self.name}' is OPEN. Request rejected."
                    )

            try:
                # Execute the original function (it must be awaitable if it's async)
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)  # type: ignore

                if current_state == CircuitState.HALF_OPEN:
                    await self._set_state(CircuitState.CLOSED)
                    await self._reset_failures()
                    logger.info(
                        "circuit_breaker_closed",
                        name=self.name,
                        mechanism="distributed",
                    )

                return result
            except Exception as e:
                failures = await self._increment_failures()
                await self._set_last_failure_time(current_time)

                if failures >= self.failure_threshold:
                    await self._set_state(
                        CircuitState.OPEN, expiry=self.recovery_timeout
                    )  # Open for recovery_timeout
                    logger.error(
                        "circuit_breaker_open",
                        name=self.name,
                        mechanism="distributed",
                        failures=failures,
                        error=str(e),
                    )

                raise e

        return wrapper


class CircuitBreakerFactory:
    """
    Factory to create appropriate circuit breaker instances based on environment.
    Supports easy switching between InMemory and Distributed (Redis) implementations.
    """

    @staticmethod
    def create(
        name: str,
        redis_client: redis.Redis | None = None,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
    ) -> Any:
        if redis_client:
            return DistributedCircuitBreaker(
                name=name,
                redis_client=redis_client,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
        return InMemoryCircuitBreaker(
            failure_threshold=failure_threshold, recovery_timeout=recovery_timeout
        )


# Global instances initialized with sensible defaults
pricing_circuit = CircuitBreakerFactory.create(
    "pricing", failure_threshold=10, recovery_timeout=60
)
db_circuit = CircuitBreakerFactory.create(
    "database", failure_threshold=5, recovery_timeout=30
)
ml_client_circuit = CircuitBreakerFactory.create(
    "ml_client", failure_threshold=5, recovery_timeout=30
)
nse_circuit = CircuitBreakerFactory.create(
    "nse", failure_threshold=3, recovery_timeout=120
)


async def initialize_circuits(redis_client: redis.Redis | None = None):
    """
    Upgrade global circuit breakers to distributed mode if Redis is available.
    """
    global pricing_circuit, db_circuit, ml_client_circuit, nse_circuit
    if redis_client:
        logger.info("upgrading_to_distributed_circuit_breakers")
        pricing_circuit = CircuitBreakerFactory.create("pricing", redis_client, 10, 60)
        db_circuit = CircuitBreakerFactory.create("database", redis_client, 5, 30)
        ml_client_circuit = CircuitBreakerFactory.create(
            "ml_client", redis_client, 5, 30
        )
        nse_circuit = CircuitBreakerFactory.create("nse", redis_client, 3, 120)
    else:
        logger.info("retaining_in_memory_circuit_breakers")
