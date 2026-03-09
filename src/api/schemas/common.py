"""
Common API Schemas (Optimized)

Shared schemas for API responses and pagination using msgspec for zero-copy performance.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    """Detailed error information."""

    message: str
    field: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    message: str
    details: list[ErrorDetail] | None = None
    request_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())


class SuccessResponse(BaseModel):
    """Standard success response."""

    message: str
    success: bool = True
    data: dict[str, Any] | None = None


class DataResponse[T](BaseModel):
    """Standard response wrapper with data field."""

    data: T
    success: bool = True
    message: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PaginationMeta(BaseModel):
    """Pagination metadata."""

    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


class PaginatedResponse[T](BaseModel):
    """Paginated response wrapper."""

    items: list[T]
    pagination: PaginationMeta

    model_config = ConfigDict(arbitrary_types_allowed=True)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())
    checks: dict[str, dict[str, Any]] = Field(default_factory=dict)
