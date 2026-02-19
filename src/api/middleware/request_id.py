"""
Request ID Middleware
=====================

Generates and tracks unique request IDs for:
- Request tracing across services
- Log correlation
- Debugging and support
"""

import secrets
import time
from collections.abc import Callable
from typing import cast

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# ... (imports stay same)


class RequestIDMiddleware(BaseHTTPMiddleware):
<<<<<<< Updated upstream
    """
    Add unique request ID to each request.

    The request ID is:
    - Generated if not provided in headers
    - Stored in request.state for access in handlers
    - Added to response headers for client correlation
    """

    HEADER_NAME = "X-Request-ID"
=======
    # ... (header constants stay same)
>>>>>>> Stashed changes

    def __init__(
        self,
        app: ASGIApp,
        header_name: str = "X-Request-ID",
        generator: Callable[[], str] | None = None,
    ):
        super().__init__(app)
        self.header_name = header_name
        # OPTIMIZED: Fast ID generator (Timestamp + Machine ID + Random)
        # machine_id = os.getenv("HOSTNAME", "node")[:4]
        self.generator = generator or (
            lambda: f"{int(time.time()*1000):x}-{secrets.token_hex(4)}"
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
<<<<<<< Updated upstream
        # Get existing request ID or generate new one
        request_id = request.headers.get(self.header_name)
=======
        # 1. Single lookup for performance
        headers = request.headers
        request_id = headers.get(self.header_name) or headers.get(self.ALT_HEADER_NAME)
>>>>>>> Stashed changes

        if not request_id:
            request_id = self.generator()

        # 2. Store in state
        request.state.request_id = request_id

        # 3. Process
        response = cast(Response, await call_next(request))

        # 4. Add to headers
        response.headers[self.header_name] = request_id

        return response


def get_request_id(request: Request) -> str | None:
    """Get request ID from request state."""
    return getattr(request.state, "request_id", None)
