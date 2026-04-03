"""
User Schemas (Optimized msgspec)

High-performance schemas for user management endpoints using msgspec for responses
and Pydantic V2 for request validation.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

import msgspec
from pydantic import BaseModel, ConfigDict, EmailStr, Field

from .common import PaginationMeta


class UserResponse(msgspec.Struct):
    """User profile response (OPTIMIZED: msgspec)."""

    id: UUID
    email: str
    full_name: str | None
    tier: str
    is_active: bool
    is_verified: bool
    is_mfa_enabled: bool
    created_at: datetime
    last_login: datetime | None = None

    @classmethod
    def from_proto(cls, proto_msg: Any) -> "UserResponse":
        """Bridge from gRPC UserInfo."""
        return cls(
            id=UUID(proto_msg.user_id),
            email=proto_msg.email,
            full_name=proto_msg.full_name or None,
            tier=proto_msg.tier,
            is_active=True,  # Assuming active if info is returned, or map from metadata
            is_verified=proto_msg.is_verified,
            is_mfa_enabled=proto_msg.mfa_enabled,
            created_at=proto_msg.created_at.to_datetime(),
            last_login=proto_msg.last_login.to_datetime()
            if proto_msg.HasField("last_login")
            else None,
        )

    def to_proto(self) -> dict[str, Any]:
        """Bridge to gRPC UserInfo."""
        return {
            "user_id": str(self.id),
            "email": self.email,
            "full_name": self.full_name or "",
            "tier": self.tier,
            "is_verified": self.is_verified,
            "mfa_enabled": self.is_mfa_enabled,
        }


class UserUpdateRequest(BaseModel):
    """User profile update request (Pydantic V2 for Validation)."""

    full_name: str | None = Field(None, max_length=255)
    email: EmailStr | None = None

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "example": {
                "full_name": "John Doe",
                "email": "john.doe@example.com",
            }
        },
    )


class UserListResponse(msgspec.Struct):
    """Paginated user list response."""

    items: list[UserResponse]
    pagination: PaginationMeta


class UserStatsResponse(msgspec.Struct):
    """User statistics response."""

    total_requests: int
    requests_today: int
    requests_this_month: int
    rate_limit_remaining: int
    rate_limit_reset: datetime


class APIKeyCreateRequest(BaseModel):
    """Request to create a new API key."""

    name: str = Field(..., min_length=1, max_length=100)

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "example": {
                "name": "Production API Key",
            }
        },
    )


class APIKeyResponse(msgspec.Struct):
    """Response containing API key metadata."""

    id: str
    name: str
    prefix: str
    created_at: datetime
    last_used_at: datetime | None = None
    raw_key: str | None = None

    @classmethod
    def from_proto(cls, proto_msg: Any) -> "APIKeyResponse":
        """Bridge from gRPC APIKeyResponse."""
        return cls(
            id=proto_msg.user_id,  # Mapping user_id as ID if that's how it's used
            name=proto_msg.key_name,
            prefix="",  # Prefix not in proto
            created_at=proto_msg.created_at.to_datetime(),
            last_used_at=None,  # Not in proto
            raw_key=None,
        )


class TierUpgradeRequest(BaseModel):
    """Tier upgrade request."""

    target_tier: str
    payment_method_id: str | None = None

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "example": {
                "target_tier": "enterprise",
                "payment_method_id": "pm_12345",
            }
        },
    )
