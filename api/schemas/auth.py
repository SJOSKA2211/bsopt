"""
Authentication Schemas (Optimized)

High-performance schemas for authentication endpoints using msgspec for responses
and Pydantic V2 for request validation.
"""

import re

import msgspec
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from src.config import settings


class LoginRequest(BaseModel):
    """User login request (Pydantic V2)."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=1, description="User password")
    remember_me: bool = Field(False, description="Extend token expiration")
    mfa_code: str | None = Field(None, description="MFA code if enabled")

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "example": {
                "email": "user@example.com",
                "password": "SecurePassword123!",
                "remember_me": False,
            }
        },
    )


class TokenResponse(BaseModel):
    """Successful token response (Pydantic)."""

    access_token: str | None = None
    refresh_token: str | None = None
    token_type: str = "bearer"
    expires_in: int | None = None
    user_id: str | None = None
    email: str | None = None
    tier: str | None = None
    requires_mfa: bool = False

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_proto(cls, proto_msg: object) -> "TokenResponse":
        """Bridge from gRPC TokenPairResponse."""
        return cls(
            access_token=getattr(proto_msg, "access_token", None),
            refresh_token=getattr(proto_msg, "refresh_token", None),
            token_type=getattr(proto_msg, "token_type", "bearer"),
            expires_in=getattr(proto_msg, "expires_in", None),
        )


# Alias for backward compatibility or specific use cases
LoginResponse = TokenResponse


class AuthResponse(BaseModel):
    """Internal authentication state response."""

    authenticated: bool
    user_id: str
    factors_verified: list[str] = []

    @classmethod
    def from_proto(cls, proto_msg: object) -> "AuthResponse":
        """Bridge from gRPC AuthResponse."""
        return cls(
            authenticated=getattr(proto_msg, "authenticated", False),
            user_id=getattr(proto_msg, "user_id", ""),
            factors_verified=list(getattr(proto_msg, "factors_verified", [])),
        )


class RegisterRequest(BaseModel):
    """User registration request (Pydantic V2)."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")
    password_confirm: str = Field(..., description="Password confirmation")
    full_name: str | None = Field(None, max_length=255, description="User's full name")
    accept_terms: bool = Field(..., description="Accept terms and conditions")

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        errors = []

        if len(v) < settings.PASSWORD_MIN_LENGTH:
            errors.append(f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters")

        if settings.PASSWORD_REQUIRE_UPPERCASE and not re.search(r"[A-Z]", v):
            errors.append("Password must contain at least one uppercase letter")

        if settings.PASSWORD_REQUIRE_LOWERCASE and not re.search(r"[a-z]", v):
            errors.append("Password must contain at least one lowercase letter")

        if settings.PASSWORD_REQUIRE_DIGIT and not re.search(r"\d", v):
            errors.append("Password must contain at least one digit")

        if settings.PASSWORD_REQUIRE_SPECIAL and not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            errors.append("Password must contain at least one special character")

        if errors:
            raise ValueError("; ".join(errors))

        return v

    @field_validator("password_confirm")
    @classmethod
    def passwords_match(cls, v: str | None, info) -> str | None:
        """Validate passwords match if confirmation is provided."""
        if v is not None and "password" in info.data and v != info.data["password"]:
            raise ValueError("Passwords do not match")
        return v

    @field_validator("accept_terms")
    @classmethod
    def must_accept_terms(cls, v: bool) -> bool:
        """Require terms acceptance."""
        if not v:
            raise ValueError("You must accept the terms and conditions")
        return v

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "example": {
                "email": "newuser@example.com",
                "password": "SecurePassword123!",
                "password_confirm": "SecurePassword123!",
                "full_name": "John Doe",
                "accept_terms": True,
            }
        },
    )


class RegisterResponse(BaseModel):
    """Successful registration response."""

    user_id: str
    email: str
    message: str
    verification_required: bool = True

    model_config = ConfigDict(from_attributes=True)


class RefreshTokenRequest(BaseModel):
    """Token refresh request."""

    refresh_token: str = Field(..., description="Current refresh token")

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={"example": {"refresh_token": "eyJhbGciOiJIUzI1NiIs..."}},
    )


class PasswordResetRequest(BaseModel):
    """Password reset request."""

    email: EmailStr = Field(..., description="Email address for password reset")

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={"example": {"email": "user@example.com"}},
    )


class PasswordResetConfirmRequest(BaseModel):
    """Password reset confirmation."""

    token: str = Field(..., description="Password reset token from email")
    new_password: str = Field(..., min_length=8, description="New password")
    new_password_confirm: str = Field(..., description="Confirm new password")

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        errors = []

        if len(v) < settings.PASSWORD_MIN_LENGTH:
            errors.append(f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters")

        if settings.PASSWORD_REQUIRE_UPPERCASE and not re.search(r"[A-Z]", v):
            errors.append("Password must contain at least one uppercase letter")

        if settings.PASSWORD_REQUIRE_LOWERCASE and not re.search(r"[a-z]", v):
            errors.append("Password must contain at least one lowercase letter")

        if settings.PASSWORD_REQUIRE_DIGIT and not re.search(r"\d", v):
            errors.append("Password must contain at least one digit")

        if errors:
            raise ValueError("; ".join(errors))

        return v

    @field_validator("new_password_confirm")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        """Validate passwords match."""
        if "new_password" in info.data and v != info.data["new_password"]:
            raise ValueError("Passwords do not match")
        return v

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "example": {
                "token": "abc123def456",
                "new_password": "NewSecurePassword123!",
                "new_password_confirm": "NewSecurePassword123!",
            }
        },
    )


class PasswordChangeRequest(BaseModel):
    """Password change request (for authenticated users)."""

    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")
    new_password_confirm: str = Field(..., description="Confirm new password")

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if len(v) < settings.PASSWORD_MIN_LENGTH:
            raise ValueError(f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters")
        return v

    @field_validator("new_password_confirm")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        """Validate passwords match."""
        if "new_password" in info.data and v != info.data["new_password"]:
            raise ValueError("Passwords do not match")
        return v

    model_config = ConfigDict(frozen=True)


class MFASetupResponse(BaseModel):
    """MFA setup response with secret and QR code."""

    secret: str
    provisioning_uri: str
    qr_code_uri: str | None = None
    backup_codes: list[str] = []

    model_config = ConfigDict(from_attributes=True)


class MFAVerifyRequest(BaseModel):
    """MFA verification request."""

    code: str = Field(..., min_length=6, max_length=8, description="TOTP code or backup code")

    @field_validator("code")
    @classmethod
    def validate_code(cls, v: str) -> str:
        """Validate code format."""
        # Remove spaces and dashes
        clean_code = v.replace(" ", "").replace("-", "")
        if not clean_code.isdigit():
            raise ValueError("Code must contain only digits")
        return clean_code

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={"example": {"code": "123456"}},
    )


class EmailVerificationRequest(BaseModel):
    """Email verification request."""

    token: str = Field(..., description="Verification token from email")

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={"example": {"token": "abc123def456"}},
    )


class WebAuthnRegistrationVerificationRequest(BaseModel):
    """WebAuthn registration verification request."""

    registration_response: dict = Field(
        ..., description="Credential object from navigator.credentials.create"
    )
    challenge: str = Field(..., description="The original challenge sent to the client")


class WebAuthnAuthenticationVerificationRequest(BaseModel):
    """WebAuthn authentication verification request."""

    authentication_response: dict = Field(
        ..., description="Credential object from navigator.credentials.get"
    )
    challenge: str = Field(..., description="The original challenge sent to the client")
    email: EmailStr = Field(..., description="User email for lookup")
    email: EmailStr = Field(..., description="User email for lookup")
