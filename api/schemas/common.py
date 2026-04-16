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


class DataResponseStruct[T](msgspec.Struct, frozen=True):
    """OPTIMIZED: Zero-copy response wrapper (msgspec)."""

    data: T
    success: bool = True
    message: str | None = None
    timestamp: datetime = msgspec.field(default_factory=lambda: datetime.utcnow())


class PaginationMetaStruct(msgspec.Struct, frozen=True):
    """OPTIMIZED: Pagination metadata (msgspec)."""

    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


class PaginatedResponseStruct[T](msgspec.Struct, frozen=True):
    """OPTIMIZED: Paginated response wrapper (msgspec)."""

    items: list[T]
    pagination: PaginationMetaStruct


class ErrorDetail(BaseModel):
    """Detailed error information (Pydantic)."""

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
    """Standard error response (Pydantic)."""

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
        if hasattr(proto_msg, "errors") and proto_msg.errors:
            details = [
                ErrorDetail(message=e.message, field=e.field, code=e.code) for e in proto_msg.errors
            ]

        return cls(
            error=proto_msg.code if hasattr(proto_msg, "code") else "INTERNAL_ERROR",
            message=proto_msg.message if hasattr(proto_msg, "message") else "Unexpected error",
            details=details,
            request_id=proto_msg.request_id if hasattr(proto_msg, "request_id") else None,
            timestamp=proto_msg.timestamp.to_datetime()
            if hasattr(proto_msg, "HasField") and proto_msg.HasField("timestamp")
            else datetime.utcnow(),
        )


class SuccessResponse(BaseModel):
    """Standard success response (Pydantic)."""

    message: str
    success: bool = True
    data: dict[str, Any] | None = None

    model_config = ConfigDict(frozen=True)


class DataResponse[T](BaseModel):
    """Standard response wrapper with data field (Pydantic)."""

    data: T
    success: bool = True
    message: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class PaginationMeta(BaseModel):
    """Pagination metadata (Pydantic)."""

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
    """Paginated response wrapper (Pydantic)."""

    items: list[T]
    pagination: PaginationMeta

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class HealthResponse(BaseModel):
    """Health check response (Pydantic)."""

    status: str
    version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())
    checks: dict[str, dict[str, Any]] = Field(default_factory=dict)

<<<<<<< HEAD
    model_config = ConfigDict(frozen=True)
=======
    model_config = ConfigDict(frozen=True)
>>>>>>> 5caa3dce9008ff117281a41908376e5ea45180e6
