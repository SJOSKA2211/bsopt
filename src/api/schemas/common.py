"""
Common API Schemas
==================

Shared schemas for API responses and pagination.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ErrorDetail(BaseModel):
    """Detailed error information."""

    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code for programmatic handling")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type or title")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[List[ErrorDetail]] = Field(None, description="Detailed error information")
    request_id: Optional[str] = Field(None, description="Request ID for support reference")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "ValidationError",
                "message": "Request validation failed",
                "details": [
                    {"field": "email", "message": "Invalid email format", "code": "invalid_format"}
                ],
                "request_id": "abc123",
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }
    )


class SuccessResponse(BaseModel):
    """Standard success response."""

    success: bool = True
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "data": {"id": "123"},
            }
        }
    )


class DataResponse(BaseModel, Generic[T]):
    """Standard response wrapper with data field."""

    success: bool = True
    data: T
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PaginationMeta(BaseModel):
    """Pagination metadata."""

    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number (1-indexed)")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper."""

    items: List[T] = Field(..., description="List of items")
    pagination: PaginationMeta = Field(..., description="Pagination metadata")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [],
                "pagination": {
                    "total": 100,
                    "page": 1,
                    "page_size": 20,
                    "total_pages": 5,
                    "has_next": True,
                    "has_prev": False,
                },
            }
        }
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    checks: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Individual component health checks"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "2.2.0",
                "timestamp": "2024-01-15T10:30:00Z",
                "checks": {
                    "database": {"status": "healthy", "latency_ms": 5},
                    "redis": {"status": "healthy", "latency_ms": 2},
                    "rabbitmq": {"status": "healthy", "latency_ms": 10},
                },
            }
        }
    )


class RateLimitInfo(BaseModel):
    """Rate limit information."""

    limit: int = Field(..., description="Maximum requests allowed")
    remaining: int = Field(..., description="Remaining requests in window")
    reset: datetime = Field(..., description="When the rate limit resets")
    window_seconds: int = Field(..., description="Rate limit window in seconds")
