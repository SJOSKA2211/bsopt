"""
Common API Schemas (Optimized msgspec)

Shared schemas for API responses and pagination using msgspec for zero-copy performance.
"""

from datetime import datetime
from typing import Any, Generic, TypeVar

import msgspec
from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class ErrorDetail(msgspec.Struct):
    """Detailed error information."""

    message: str
    field: str | None = None
    code: str | None = None


class ErrorResponse(msgspec.Struct):
    """Standard error response."""

    error: str
    message: str
    details: list[ErrorDetail] | None = None
    request_id: str | None = None
    timestamp: datetime = msgspec.field(default_factory=datetime.utcnow)


class SuccessResponse(msgspec.Struct):
    """Standard success response."""

    message: str
    success: bool = True
    data: dict[str, Any] | None = None


class DataResponse(BaseModel, Generic[T]):
    """Standard response wrapper with data field."""

    data: T
    success: bool = True
    message: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())

    model_config = ConfigDict(arbitrary_types_allowed=True)


from pydantic import Field


class PaginationMeta(msgspec.Struct):
    """Pagination metadata."""

    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


class PaginatedResponse(msgspec.Struct):
    """Paginated response wrapper."""

    items: list[Any]
    pagination: PaginationMeta


class HealthResponse(msgspec.Struct):
    """Health check response."""

    status: str
    version: str
    timestamp: datetime = msgspec.field(default_factory=datetime.utcnow)
    checks: dict[str, dict[str, Any]] = msgspec.field(default_factory=dict)
