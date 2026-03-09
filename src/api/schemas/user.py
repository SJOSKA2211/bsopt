"""
User Schemas (Optimized msgspec)

High-performance schemas for user management endpoints using msgspec for responses
and Pydantic for request validation.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

import msgspec
from pydantic import BaseModel, ConfigDict, EmailStr, Field

from .common import PaginationMeta


class UserResponse(msgspec.Struct):
    """User profile response."""

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
    def from_orm(cls, user: Any) -> "UserResponse":
        return cls(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            tier=user.tier,
            is_active=user.is_active,
            is_verified=user.is_verified,
            is_mfa_enabled=user.is_mfa_enabled,
            created_at=user.created_at,
            last_login=user.last_login,
        )


class UserUpdateRequest(BaseModel):
    """User profile update request (Pydantic for Validation)."""

    full_name: str | None = Field(None, max_length=255)
    email: EmailStr | None = None

    model_config = ConfigDict(extra="forbid")


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


class APIKeyResponse(msgspec.Struct):
    """Response containing API key metadata."""

    id: str
    name: str
    prefix: str
    created_at: datetime
    last_used_at: datetime | None = None
    raw_key: str | None = None


class TierUpgradeRequest(BaseModel):
    """Tier upgrade request."""

    target_tier: str
    payment_method_id: str | None = None
