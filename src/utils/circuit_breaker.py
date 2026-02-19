import asyncio
import time
from collections.abc import Callable
from enum import Enum
from functools import wraps
from typing import Any

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

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
    ):
        self.name = name
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
        logger.info("circuit_breaker_reset", name=self.name, mechanism="in_memory")

    def __call__(self, func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info(
                        "circuit_breaker_half_open",
                        name=self.name,
                        mechanism="in_memory",
                    )
                else:
                    raise Exception(
                        f"Circuit Breaker '{self.name}' is OPEN. Request rejected."
                    )

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    # Offload sync function to thread to avoid blocking loop if needed,
                    # but simple call for now.
                    result = func(*args, **kwargs)

                if self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info(
                        "circuit_breaker_closed", name=self.name, mechanism="in_memory"
                    )

                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.error(
                        "circuit_breaker_open",
                        name=self.name,
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
    Distributed Circuit Breaker with OPTIMIZED Lua atomicity.
    Reduces Redis roundtrips and ensures thread-safe state transitions.
    """

    CHECK_LUA = """
    local state = redis.call('GET', KEYS[1]) or 'CLOSED'
    local failures = tonumber(redis.call('GET', KEYS[2]) or '0')
    local last_failure = tonumber(redis.call('GET', KEYS[3]) or '0')
    local now = tonumber(ARGV[1])
    local recovery_timeout = tonumber(ARGV[2])
    
    if state == 'OPEN' then
        if now - last_failure > recovery_timeout then
            redis.call('SET', KEYS[1], 'HALF_OPEN')
            return 'HALF_OPEN'
        end
        return 'OPEN'
    end
    return state
    """

    def __init__(
        self,
        name: str,
        redis_client: redis.Redis,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
    ):
        self.name = name
        self.redis_client = redis_client
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.keys = [
            f"{name}:cb_state",
            f"{name}:cb_failures",
            f"{name}:cb_last_failure",
        ]
        self._check_script = redis_client.register_script(self.CHECK_LUA)

    def __call__(self, func: Callable):
        from anyio.to_thread import run_sync

        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = int(time.time())

            # OPTIMIZED: Atomic state check via Lua
            current_state = await self._check_script(
                keys=self.keys, args=[now, self.recovery_timeout]
            )
            current_state = (
                current_state.decode()
                if isinstance(current_state, bytes)
                else current_state
            )

            if current_state == "OPEN":
                raise Exception(f"Circuit Breaker '{self.name}' is OPEN.")

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    # OPTIMIZED: Offload sync function to avoid loop stalling
                    result = await run_sync(func, *args, **kwargs)

                if current_state == "HALF_OPEN":
                    await self.redis_client.delete(self.keys[0], self.keys[1])  # Reset
                    logger.info("circuit_breaker_closed", name=self.name)

                return result
            except Exception as e:
                # Increment failures and update timestamp
                await self.redis_client.incr(self.keys[1])
                await self.redis_client.set(self.keys[2], now)

                failures = int(await self.redis_client.get(self.keys[1]) or 0)
                if failures >= self.failure_threshold:
                    await self.redis_client.set(
                        self.keys[0], "OPEN", ex=self.recovery_timeout
                    )
                    logger.error(
                        "circuit_breaker_opened", name=self.name, failures=failures
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
webhook_circuit = CircuitBreakerFactory.create(
    "webhook", failure_threshold=5, recovery_timeout=30
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
