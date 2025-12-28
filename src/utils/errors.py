"""
Robust Error Handling Utilities
"""

import functools
import logging
import time
import traceback
from typing import Any, Callable

from fastapi import HTTPException, status

logger = logging.getLogger("audit")


def robust_pricing_task(error_return_value: Any = None):
    """
    Decorator for pricing tasks to ensure they never crash the worker
    and log comprehensive error details.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                duration = (time.perf_counter() - start_time) * 1000
                logger.error(
                    f"Pricing Task Failure: {func.__name__}\n"
                    f"Error: {str(e)}\n"
                    f"Duration: {duration:.2f}ms\n"
                    f"Traceback: {traceback.format_exc()}",
                    extra={"task": func.__name__, "error": str(e), "duration_ms": duration},
                )
                return error_return_value

        return wrapper

    return decorator


class ServiceUnavailableException(HTTPException):
    def __init__(self, service_name: str):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service '{service_name}' is currently unavailable. Please try again later.",
        )
