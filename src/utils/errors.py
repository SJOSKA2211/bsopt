"""
Robust Error Handling Utilities
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from typing import Any

from src.shared.off_heap_logger import omega_logger

logger = logging.getLogger("audit")


def robust_pricing_task(error_return_value: Any = None):
    """
    OPTIMIZED: Async-aware decorator for fail-safe task execution.
    Logs to high-speed off-heap buffer to prevent I/O blocking.
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    ms = (time.perf_counter() - start) * 1000
                    omega_logger.log("pricing_task_failed", task=func.__name__, error=str(e), ms=ms)
                    return error_return_value
            return wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    ms = (time.perf_counter() - start) * 1000
                    omega_logger.log("pricing_task_failed_sync", task=func.__name__, error=str(e), ms=ms)
                    return error_return_value
            return wrapper
    return decorator
