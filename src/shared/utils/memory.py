import functools
import os
import time
from collections.abc import Callable
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

def profile_memory(func: Callable) -> Callable:
    """
    Decorator to profile memory and execution time of a function.
    Helps in detecting GPU/CPU memory leaks in mathematical kernels.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        import psutil

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        duration = (time.perf_counter() - start_time) * 1000

        mem_after = process.memory_info().rss / (1024 * 1024)
        mem_diff = mem_after - mem_before

        logger.info(
            "function_memory_profile",
            function=func.__name__,
            duration_ms=round(duration, 2),
            mem_before_mb=round(mem_before, 2),
            mem_after_mb=round(mem_after, 2),
            mem_diff_mb=round(mem_diff, 2),
        )

        return result

    return wrapper

def profile_gpu_memory(func: Callable) -> Callable:
    """
    Decorator to profile CUDA GPU memory.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            from numba import cuda

            cuda.get_current_device()
            # Simple profile using free/total memory
            free, total = cuda.current_context().get_memory_info()
            mem_before_mb = (total - free) / (1024 * 1024)

            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            duration = (time.perf_counter() - start_time) * 1000

            free, total = cuda.current_context().get_memory_info()
            mem_after_mb = (total - free) / (1024 * 1024)

            logger.info(
                "gpu_memory_profile",
                function=func.__name__,
                duration_ms=round(duration, 2),
                gpu_mem_before_mb=round(mem_before_mb, 2),
                gpu_mem_after_mb=round(mem_after_mb, 2),
                gpu_mem_diff_mb=round(mem_after_mb - mem_before_mb, 2),
            )
            return result
        except (ImportError, Exception):
            # Fallback to CPU profiling if CUDA is not available
            return profile_memory(func)(*args, **kwargs)

    return wrapper
