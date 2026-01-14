"""
API Schemas
===========

Pydantic models for request/response validation:
- Authentication schemas
- User management schemas
- Pricing schemas
- Error responses
"""

from .auth import (
    LoginRequest,
    LoginResponse,
    MFASetupResponse,
    MFAVerifyRequest,
    PasswordChangeRequest,
    PasswordResetConfirm,
    PasswordResetRequest,
    RefreshTokenRequest,
    RegisterRequest,
    RegisterResponse,
    TokenResponse,
)
from .common import (
    ErrorResponse,
    HealthResponse,
    PaginatedResponse,
    SuccessResponse,
)
from .pricing import (
    BatchPriceRequest,
    BatchPriceResponse,
    GreeksRequest,
    GreeksResponse,
    PriceRequest,
    PriceResponse,
)
from .user import (
    UserListResponse,
    UserResponse,
    UserUpdateRequest,
)

__all__ = [
    # Auth
    "LoginRequest",
    "LoginResponse",
    "RegisterRequest",
    "RegisterResponse",
    "TokenResponse",
    "RefreshTokenRequest",
    "PasswordResetRequest",
    "PasswordResetConfirm",
    "PasswordChangeRequest",
    "MFASetupResponse",
    "MFAVerifyRequest",
    # User
    "UserResponse",
    "UserUpdateRequest",
    "UserListResponse",
    # Pricing
    "PriceRequest",
    "PriceResponse",
    "BatchPriceRequest",
    "BatchPriceResponse",
    "GreeksRequest",
    "GreeksResponse",
    # Common
    "ErrorResponse",
    "SuccessResponse",
    "PaginatedResponse",
    "HealthResponse",
]
