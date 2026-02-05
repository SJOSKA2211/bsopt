"""
Unified Resilience and Retry Utilities
======================================
Provides standard decorators for robust error handling and backoff.
"""

import asyncio
import functools
import random
from collections.abc import Callable

import structlog

logger = structlog.get_logger(__name__)

def retry_with_backoff(
    retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: type[Exception] | tuple[type[Exception], ...] = (Exception,)
):
    """
    Standardized retry decorator with exponential backoff and jitter.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == retries:
                        logger.error(
                            "retry_limit_reached", 
                            func=func.__name__, 
                            attempts=attempt+1, 
                            error=str(e)
                        )
                        raise
                    
                    wait_time = delay * (backoff_factor ** attempt)
                    if jitter:
                        wait_time *= (0.5 + random.random()) # nosec B311
                    
                    logger.warning(
                        "retrying_operation", 
                        func=func.__name__, 
                        attempt=attempt+1, 
                        wait_time=round(wait_time, 2),
                        error=str(e)
                    )
                    await asyncio.sleep(wait_time)
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
