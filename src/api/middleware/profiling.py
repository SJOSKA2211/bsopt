"""
API Performance Profiling Middleware
"""

import logging
import time

import pyinstrument
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import HTMLResponse

logger = logging.getLogger(__name__)


class ProfilingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that profiles requests and logs slow ones.
    If 'profile=true' is in the query params, it returns a pyinstrument HTML report.
    """

    async def dispatch(self, request: Request, call_next):
        from src.config import settings

        should_profile = request.query_params.get("profile") == "true" and settings.DEBUG

        if should_profile:
            profiler = pyinstrument.Profiler(interval=0.001)
            profiler.start()

            response = await call_next(request)

            profiler.stop()
            return HTMLResponse(profiler.output_html())

        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = (time.perf_counter() - start_time) * 1000

        if process_time > 100:  # Log requests slower than 100ms
            logger.warning(
                f"Slow request: {request.method} {request.url.path} took {process_time:.2f}ms"
            )

        response.headers["X-Process-Time-MS"] = f"{process_time:.2f}"
        return response
