"""
Request ID Middleware
=====================

Generates and tracks unique request IDs for:
- Request tracing across services
- Log correlation
- Debugging and support
"""

import uuid
from collections.abc import Callable
from typing import cast

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Add unique request ID to each request.

    The request ID is:
    - Generated if not provided in headers
    - Stored in request.state for access in handlers
    - Added to response headers for client correlation
    """

    HEADER_NAME = "X-Request-ID"

    def __init__(
        self,
        app: ASGIApp,
        header_name: str = "X-Request-ID",
        generator: Callable[[], str] | None = None,
    ):
        super().__init__(app)
        self.header_name = header_name
        self.generator = generator or (lambda: str(uuid.uuid4()))

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get existing request ID or generate new one
        request_id = request.headers.get(self.header_name)

        if not request_id:
            request_id = self.generator()

        # Store in request state
        request.state.request_id = request_id

        # Process request
        response = cast(Response, await call_next(request))

        # Add to response headers
        response.headers[self.header_name] = request_id

        return response


def get_request_id(request: Request) -> str | None:
    """Get request ID from request state."""
    return getattr(request.state, "request_id", None)
