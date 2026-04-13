import asyncio
import time
from collections.abc import Callable
from enum import Enum
from functools import wraps

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
                    raise Exception(f"Circuit Breaker '{self.name}' is OPEN. Request rejected.")

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
                    logger.info("circuit_breaker_closed", name=self.name, mechanism="in_memory")

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
        self._failure_count_cache = 0

    def __call__(self, func: Callable):
        from anyio.to_thread import run_sync

        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = int(time.time())

            current_state = await self._check_script(
                keys=self.keys, args=[now, self.recovery_timeout]
            )
            current_state = (
                current_state.decode() if isinstance(current_state, bytes) else current_state
            )

            if current_state == "OPEN":
                raise Exception(f"Circuit Breaker '{self.name}' is OPEN.")

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = await run_sync(func, *args, **kwargs)

                if current_state == "HALF_OPEN":
                    await self.redis_client.delete(self.keys[0], self.keys[1])  # Reset
                    self._failure_count_cache = 0
                    logger.info("circuit_breaker_closed", name=self.name)

                # Periodically sync failure count from Redis (e.g., every 5 calls or if cache is 0)
                if now % 5 == 0 or self._failure_count_cache == 0:
                    val = await self.redis_client.get(self.keys[1])
                    self._failure_count_cache = int(val) if val else 0

                return result
            except Exception as e:
                # Increment failures and update timestamp
                await self.redis_client.incr(self.keys[1])
                await self.redis_client.set(self.keys[2], now)
                self._failure_count_cache += 1

                failures = int(await self.redis_client.get(self.keys[1]) or 0)
                self._failure_count_cache = failures
                if failures >= self.failure_threshold:
                    await self.redis_client.set(self.keys[0], "OPEN", ex=self.recovery_timeout)
                    logger.error("circuit_breaker_opened", name=self.name, failures=failures)

                raise e

        return wrapper

    @property
    def failure_count(self):
        return self._failure_count_cache


class CircuitBreakerProxy:
    """
    Proxy for Circuit Breakers to allow dynamic upgrading from In-Memory to Distributed
    without breaking established module-level imports.
    """

    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.name = name
        self._cb = InMemoryCircuitBreaker(name, failure_threshold, recovery_timeout)

    def upgrade(
        self,
        redis_client: redis.Redis,
        failure_threshold: int | None = None,
        recovery_timeout: int | None = None,
    ):
        """Upgrades the internal implementation to DistributedCircuitBreaker."""
        ft = failure_threshold or self._cb.failure_threshold
        rt = recovery_timeout or self._cb.recovery_timeout
        logger.info("upgrading_circuit_breaker_to_distributed", name=self.name)
        self._cb = DistributedCircuitBreaker(self.name, redis_client, ft, rt)

    def reset(self):
        """Resets the internal circuit breaker state."""
        if hasattr(self._cb, "reset"):
            self._cb.reset()

    def __call__(self, func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self._cb(func)(*args, **kwargs)

        return wrapper

    @property
    def state(self):
        return getattr(self._cb, "state", "UNKNOWN")

    @property
    def failure_count(self):
        if isinstance(self._cb, InMemoryCircuitBreaker):
            return self._cb.failure_count
        if isinstance(self._cb, DistributedCircuitBreaker):
            return self._cb.failure_count
        return -1


# Global instances initialized as Proxies
pricing_circuit = CircuitBreakerProxy("pricing", failure_threshold=10, recovery_timeout=60)
db_circuit = CircuitBreakerProxy("database", failure_threshold=5, recovery_timeout=30)
ml_client_circuit = CircuitBreakerProxy("ml_client", failure_threshold=5, recovery_timeout=30)
nse_circuit = CircuitBreakerProxy("nse", failure_threshold=3, recovery_timeout=120)
webhook_circuit = CircuitBreakerProxy("webhook", failure_threshold=5, recovery_timeout=30)


async def initialize_circuits(redis_client: redis.Redis | None = None):
    """
    Upgrade global circuit breaker proxies to distributed mode if Redis is available.
    """
    if redis_client:
        logger.info("initializing_distributed_circuits")
        pricing_circuit.upgrade(redis_client, 10, 60)
        db_circuit.upgrade(redis_client, 5, 30)
        ml_client_circuit.upgrade(redis_client, 5, 30)
        nse_circuit.upgrade(redis_client, 3, 120)
        webhook_circuit.upgrade(redis_client, 5, 30)
    else:
        logger.info("retaining_in_memory_circuits")