"""
Common API Schemas (Optimized)

Shared schemas for API responses and pagination using msgspec for zero-copy performance
and Pydantic V2 for request validation and complex logic.
"""

from datetime import datetime
from typing import Any, TypeVar

import msgspec
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class DataResponseStruct(msgspec.Struct):
    """OPTIMIZED: Zero-copy response wrapper."""

    data: Any
    success: bool = True
    message: str | None = None
    timestamp: datetime = msgspec.field(default_factory=lambda: datetime.utcnow())


class PaginationMetaStruct(msgspec.Struct):
    """OPTIMIZED: Pagination metadata."""

    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


class PaginatedResponseStruct(msgspec.Struct):
    """OPTIMIZED: Paginated response wrapper."""

    items: list[Any]
    pagination: PaginationMetaStruct


class ErrorDetail(BaseModel):
    """Detailed error information."""

    message: str
    field: str | None = None
    code: str | None = None

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "example": {
                "message": "Field is required",
                "field": "email",
                "code": "missing_field",
            }
        },
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    message: str
    details: list[ErrorDetail] | None = None
    request_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "example": {
                "error": "Validation Error",
                "message": "One or more fields failed validation",
                "details": [
                    {"message": "Invalid email format", "field": "email", "code": "invalid_format"}
                ],
                "request_id": "req_123456",
            }
        },
    )

    @classmethod
    def from_proto(cls, proto_msg: Any) -> "ErrorResponse":
        """Bridge from gRPC ErrorResponse."""
        details = None
        if proto_msg.errors:
            details = [
                ErrorDetail(message=e.message, field=e.field, code=e.code) for e in proto_msg.errors
            ]

        return cls(
            error=proto_msg.code,
            message=proto_msg.message,
            details=details,
            request_id=proto_msg.request_id,
            timestamp=proto_msg.timestamp.to_datetime()
            if proto_msg.HasField("timestamp")
            else datetime.utcnow(),
        )


class SuccessResponse(BaseModel):
    """Standard success response."""

    message: str
    success: bool = True
    data: dict[str, Any] | None = None

    model_config = ConfigDict(frozen=True)


class DataResponse[T](BaseModel):
    """Standard response wrapper with data field."""

    data: T
    success: bool = True
    message: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class PaginationMeta(BaseModel):
    """Pagination metadata."""

    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool

    model_config = ConfigDict(frozen=True)

    @classmethod
    def from_proto(cls, proto_msg: Any) -> "PaginationMeta":
        """Bridge from gRPC PaginationResponse."""
        return cls(
            total=proto_msg.total_items,
            page=proto_msg.current_page,
            page_size=proto_msg.page_size,
            total_pages=proto_msg.total_pages,
            has_next=proto_msg.has_next,
            has_prev=proto_msg.has_previous,
        )


class PaginatedResponse[T](BaseModel):
    """Paginated response wrapper."""

    items: list[T]
    pagination: PaginationMeta

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())
    checks: dict[str, dict[str, Any]] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)
