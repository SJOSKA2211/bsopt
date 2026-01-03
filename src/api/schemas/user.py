"""
User Schemas
============

Pydantic models for user management endpoints.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from .common import PaginationMeta


class UserResponse(BaseModel):
    """User profile response."""

    id: UUID = Field(..., description="User ID")
    email: EmailStr = Field(..., description="User email")
    full_name: Optional[str] = Field(None, description="User's full name")
    tier: str = Field(..., description="Subscription tier")
    is_active: bool = Field(..., description="Account active status")
    is_verified: bool = Field(..., description="Email verified status")
    is_mfa_enabled: bool = Field(..., description="MFA enabled status")
    created_at: datetime = Field(..., description="Account creation date")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "email": "user@example.com",
                "full_name": "John Doe",
                "tier": "pro",
                "is_active": True,
                "is_verified": True,
                "is_mfa_enabled": False,
                "created_at": "2024-01-01T00:00:00Z",
                "last_login": "2024-01-15T10:30:00Z",
            }
        }
    )


class UserUpdateRequest(BaseModel):
    """User profile update request."""

    full_name: Optional[str] = Field(None, max_length=255, description="User's full name")
    email: Optional[EmailStr] = Field(None, description="New email address")

    model_config = ConfigDict(
        json_schema_extra={"example": {"full_name": "John Smith"}}
    )


class UserListResponse(BaseModel):
    """Paginated user list response."""

    items: List[UserResponse] = Field(..., description="List of users")
    pagination: PaginationMeta = Field(..., description="Pagination metadata")


class UserStatsResponse(BaseModel):
    """User statistics response."""

    total_requests: int = Field(..., description="Total API requests")
    requests_today: int = Field(..., description="Requests made today")
    requests_this_month: int = Field(..., description="Requests made this month")
    rate_limit_remaining: int = Field(..., description="Remaining requests in current window")
    rate_limit_reset: datetime = Field(..., description="When rate limit resets")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_requests": 15000,
                "requests_today": 250,
                "requests_this_month": 3500,
                "rate_limit_remaining": 750,
                "rate_limit_reset": "2024-01-15T11:00:00Z",
            }
        }
    )


class APIKeyCreateRequest(BaseModel):
    """Request to create a new API key."""
    name: str = Field(..., min_length=1, max_length=100, description="Friendly name for the key")


class APIKeyResponse(BaseModel):
    """Response containing API key metadata."""
    id: str
    name: str
    prefix: str
    created_at: datetime
    last_used_at: Optional[datetime] = None
    raw_key: Optional[str] = None # Only populated on creation


class TierUpgradeRequest(BaseModel):
    """Tier upgrade request."""

    target_tier: str = Field(..., description="Target subscription tier")
    payment_method_id: Optional[str] = Field(None, description="Payment method ID")

    model_config = ConfigDict(
        json_schema_extra={"example": {"target_tier": "pro", "payment_method_id": "pm_123456789"}}
    )
