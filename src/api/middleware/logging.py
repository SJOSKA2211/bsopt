"""
Request Logging Middleware
==========================

Comprehensive request/response logging with:
- Structured JSON logging
- Database persistence for audit trails
- Performance metrics
- Error tracking
- Sensitive data redaction
"""

import logging
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

import msgspec
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Dedicated logger for request logs (can be configured separately)
request_logger = logging.getLogger("requests")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Log all HTTP requests and responses.

    Features:
    - Structured JSON logging
    - Request/response timing
    - Error capture with stack traces
    - Sensitive data redaction
    - Configurable log levels by path
    - Optional database persistence
    """

    # Headers to redact
    SENSITIVE_HEADERS: set[str] = {
        "authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "x-auth-token",
        "x-csrf-token",
        "proxy-authorization",
    }

    # Query params to redact
    SENSITIVE_PARAMS: set[str] = {
        "password",
        "token",
        "api_key",
        "secret",
        "access_token",
        "refresh_token",
    }

    # Paths to skip logging (high-frequency, low-value)
    SKIP_PATHS: set[str] = {
        "/health",
        "/metrics",
        "/favicon.ico",
    }

    # Paths with reduced logging (only errors)
    REDUCED_LOG_PATHS: set[str] = {
        "/docs",
        "/redoc",
        "/openapi.json",
    }

    def __init__(
        self,
        app: ASGIApp,
        log_request_body: bool = False,
        log_response_body: bool = False,
        max_body_length: int = 1000,
        persist_to_db: bool = True,
        slow_request_threshold_ms: int = 1000,
    ):
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.max_body_length = max_body_length
        self.persist_to_db = persist_to_db
        self.slow_request_threshold_ms = slow_request_threshold_ms

    def _redact_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Redact sensitive headers."""
        redacted = {}
        for key, value in headers.items():
            if key.lower() in self.SENSITIVE_HEADERS:
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = value
        return redacted

    def _redact_params(self, params: dict[str, str]) -> dict[str, str]:
        """Redact sensitive query parameters."""
        redacted = {}
        for key, value in params.items():
            if key.lower() in self.SENSITIVE_PARAMS:
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = value
        return redacted

    def _truncate_body(self, body: str) -> str:
        """Truncate body if too long."""
        if len(body) > self.max_body_length:
            return (
                body[: self.max_body_length]
                + f"... [truncated, {len(body)} bytes total]"
            )
        return body

    def _get_client_ip(self, request: Request) -> str:
        """Get real client IP."""
        # Check state first (set by IPBlockMiddleware)
        if hasattr(request.state, "client_ip"):
            return cast(str, request.state.client_ip)

        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()

        return cast(str, request.client.host if request.client else "unknown")

    def _get_user_info(self, request: Request) -> dict[str, Any]:
        """Extract user info from consolidated request state."""
        return {
            "user_id": getattr(request.state, "user_id", None),
            "user_email": getattr(request.state, "user_email", None),
            "user_tier": getattr(request.state, "user_tier", None),
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        if self._should_skip(path):
            return cast(Response, await call_next(request))

        # Start timing
        start_time = time.perf_counter()
        
        # 1. Use existing request ID from my optimized generator
        request_id = getattr(request.state, "request_id", "unknown")

        # ... (process request)
        response = await call_next(request)
        status_code = response.status_code
        
        # 2. Optimized Logging with Sampling
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        from src.shared.observability import settings
        
        # Use sampling for successful logs to reduce I/O
        should_log = True
        if 200 <= status_code < 300:
            import random
            if random.random() > getattr(settings, "LOG_SAMPLING_RATE", 0.1):
                should_log = False
        
        if should_log or status_code >= 400:
            log_entry = {
                "request_id": request_id,
                "method": request.method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration_ms,
                **self._get_user_info(request)
            }
            
            # OPTIMIZED: Hand off to Off-Heap Logger for zero-latency persistence
            from src.shared.off_heap_logger import omega_logger
            omega_logger.log("api_request", **log_entry)
            
            # Still log to console for visibility
            request_logger.info(msgspec.json.encode(log_entry).decode("utf-8"))

        return cast(Response, response)

        return cast(Response, response)

    async def _persist_log(self, log_entry: dict[str, Any], request: Request) -> None:
        """Persist log entry to database using anyio.to_thread to avoid blocking."""
        from anyio.to_thread import run_sync

        def _save():
            try:
                from src.database import SessionLocal
                from src.database.models import RequestLog

                session = SessionLocal()
                try:
                    # OPTIMIZED: Convert query_params dict to string using msgspec
                    query_params_str = None
                    if log_entry.get("query_params"):
                        query_params_str = msgspec.json.encode(
                            log_entry["query_params"]
                        ).decode("utf-8")

                    request_log = RequestLog(
                        request_id=log_entry["request_id"],
                        method=log_entry["method"],
                        path=log_entry["path"][:500],
                        query_params=query_params_str,
                        headers=log_entry.get("headers"),
                        client_ip=log_entry["client_ip"],
                        user_id=(
                            UUID(log_entry["user_id"])
                            if log_entry.get("user_id")
                            else None
                        ),
                        status_code=log_entry["status_code"],
                        response_time_ms=log_entry["duration_ms"],
                        error_message=log_entry.get("error", {}).get("message"),
                    )
                    session.add(request_log)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    logger.error(f"Database persistence failed: {e}")
                finally:
                    session.close()
            except Exception as e:
                logger.error(f"Database persistence failed: {e}")

        await run_sync(_save)


class StructuredLogger:
    """
    Structured logging utility for consistent log format.

    Provides methods for logging with standard fields
    that integrate with log aggregation systems.
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.default_fields: dict[str, Any] = {}

    def set_default_fields(self, **fields):
        """Set default fields to include in all logs."""
        self.default_fields.update(fields)

    def _build_log_entry(self, message: str, level: str, **kwargs) -> dict[str, Any]:
        """Build structured log entry."""
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "message": message,
            **self.default_fields,
            **kwargs,
        }
        return entry

    def debug(self, message: str, **kwargs):
        entry = self._build_log_entry(message, "DEBUG", **kwargs)
        self.logger.debug(msgspec.json.encode(entry).decode("utf-8"))

    def info(self, message: str, **kwargs):
        entry = self._build_log_entry(message, "INFO", **kwargs)
        self.logger.info(msgspec.json.encode(entry).decode("utf-8"))

    def warning(self, message: str, **kwargs):
        entry = self._build_log_entry(message, "WARNING", **kwargs)
        self.logger.warning(msgspec.json.encode(entry).decode("utf-8"))

    def error(self, message: str, **kwargs):
        entry = self._build_log_entry(message, "ERROR", **kwargs)
        self.logger.error(msgspec.json.encode(entry).decode("utf-8"))

    def critical(self, message: str, **kwargs):
        entry = self._build_log_entry(message, "CRITICAL", **kwargs)
        self.logger.critical(msgspec.json.encode(entry).decode("utf-8"))

    def exception(self, message: str, exc_info=True, **kwargs):
        if exc_info:
            kwargs["traceback"] = traceback.format_exc()
        entry = self._build_log_entry(message, "ERROR", **kwargs)
        self.logger.error(msgspec.json.encode(entry).decode("utf-8"))
