"""
Unified Resilience and Retry Utilities
Provides standard decorators for robust error handling and backoff.
"""

import asyncio
import functools
import random
from collections.abc import Callable
from typing import Any, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: type[Exception] | tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Standardized retry decorator with exponential backoff and jitter.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None

            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == retries:
                        break

                    # Exponential Backoff with Jitter
                    wait_time = initial_delay * (backoff_factor**attempt)
                    if jitter:
                        wait_time *= 0.5 + random.random()  # nosec B311

                    logger.warning(
                        "retrying_operation",
                        func=func.__name__,
                        attempt=attempt + 1,
                        wait_time=f"{wait_time:.2f}s",
                        error=str(e),
                    )
                    await asyncio.sleep(wait_time)

            logger.error(
                "retry_limit_reached",
                func=func.__name__,
                attempts=retries + 1,
                error=str(last_exception),
            )
            if last_exception:
                raise last_exception
            raise RuntimeError("Retry failed without exception")

        return wrapper

    return decorator
