"""
Authentication Schemas
======================

Pydantic models for authentication endpoints.
"""

import re

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from src.config import settings


class LoginRequest(BaseModel):
    """User login request."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=1, description="User password") # nosec B105 B106
    remember_me: bool = Field(False, description="Extend token expiration") # nosec B105 B106
    mfa_code: str | None = Field(None, description="MFA code if enabled")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "user@example.com",
                "password": "SecurePassword123!", # nosec B105 B106
                "remember_me": False,
            }
        }
    )


class LoginResponse(BaseModel):
    """Successful login response."""

    access_token: str = Field(..., description="JWT access token") # nosec B105 B106
    refresh_token: str = Field(..., description="JWT refresh token") # nosec B105 B106
    token_type: str = "bearer" # nosec B105 B106
    expires_in: int = Field(..., description="Access token expiration in seconds") # nosec B105 B106
    user_id: str = Field(..., description="User ID")
    email: str = Field(..., description="User email")
    tier: str = Field(..., description="User subscription tier")
    requires_mfa: bool = False

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIs...", # nosec B105 B106
                "refresh_token": "eyJhbGciOiJIUzI1NiIs...", # nosec B105 B106
                "token_type": "bearer", # nosec B105 B106
                "expires_in": 1800,
                "user_id": "123e4567-e89b-12d3-a456-426614174000",
                "email": "user@example.com",
                "tier": "pro",
                "requires_mfa": False,
            }
        }
    )


class RegisterRequest(BaseModel):
    """User registration request."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password") # nosec B105 B106
    password_confirm: str = Field(..., description="Password confirmation") # nosec B105 B106
    full_name: str | None = Field(None, max_length=255, description="User's full name")
    accept_terms: bool = Field(..., description="Accept terms and conditions")

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str: # nosec B105 B106
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
    def passwords_match(cls, v: str, info) -> str: # nosec B105 B106
        """Validate passwords match."""
        if "password" in info.data and v != info.data["password"]: # nosec B105 B106
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
        json_schema_extra={
            "example": {
                "email": "newuser@example.com",
                "password": "SecurePassword123!", # nosec B105 B106
                "password_confirm": "SecurePassword123!", # nosec B105 B106
                "full_name": "John Doe",
                "accept_terms": True,
            }
        }
    )


class RegisterResponse(BaseModel):
    """Successful registration response."""

    user_id: str = Field(..., description="Created user ID")
    email: str = Field(..., description="User email")
    message: str = Field(..., description="Success message")
    verification_required: bool = Field(True, description="Email verification required")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "123e4567-e89b-12d3-a456-426614174000",
                "email": "newuser@example.com",
                "message": "Registration successful. Please verify your email.",
                "verification_required": True,
            }
        }
    )


class TokenResponse(BaseModel):
    """Token response (for refresh)."""

    access_token: str = Field(..., description="New JWT access token") # nosec B105 B106
    refresh_token: str = Field(..., description="New JWT refresh token") # nosec B105 B106
    token_type: str = Field("bearer", description="Token type") # nosec B105 B106
    expires_in: int = Field(..., description="Access token expiration in seconds") # nosec B105 B106


class RefreshTokenRequest(BaseModel):
    """Token refresh request."""

    refresh_token: str = Field(..., description="Current refresh token") # nosec B105 B106

    model_config = ConfigDict(
        json_schema_extra={"example": {"refresh_token": "eyJhbGciOiJIUzI1NiIs..."}} # nosec B105 B106
    )


class PasswordResetRequest(BaseModel):
    """Password reset request."""

    email: EmailStr = Field(..., description="Email address for password reset") # nosec B105 B106

    model_config = ConfigDict(
        json_schema_extra={"example": {"email": "user@example.com"}}
    )


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation."""

    token: str = Field(..., description="Password reset token from email") # nosec B105 B106
    new_password: str = Field(..., min_length=8, description="New password") # nosec B105 B106
    new_password_confirm: str = Field(..., description="Confirm new password") # nosec B105 B106

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: str) -> str: # nosec B105 B106
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
    def passwords_match(cls, v: str, info) -> str: # nosec B105 B106
        """Validate passwords match."""
        if "new_password" in info.data and v != info.data["new_password"]: # nosec B105 B106
            raise ValueError("Passwords do not match")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "token": "abc123def456", # nosec B105 B106
                "new_password": "NewSecurePassword123!", # nosec B105 B106
                "new_password_confirm": "NewSecurePassword123!", # nosec B105 B106
            }
        }
    )


class PasswordChangeRequest(BaseModel):
    """Password change request (for authenticated users)."""

    current_password: str = Field(..., description="Current password") # nosec B105 B106
    new_password: str = Field(..., min_length=8, description="New password") # nosec B105 B106
    new_password_confirm: str = Field(..., description="Confirm new password") # nosec B105 B106

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: str) -> str: # nosec B105 B106
        """Validate password strength."""
        if len(v) < settings.PASSWORD_MIN_LENGTH:
            raise ValueError(f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters")
        return v

    @field_validator("new_password_confirm")
    @classmethod
    def passwords_match(cls, v: str, info) -> str: # nosec B105 B106
        """Validate passwords match."""
        if "new_password" in info.data and v != info.data["new_password"]: # nosec B105 B106
            raise ValueError("Passwords do not match")
        return v


class MFASetupResponse(BaseModel):
    """MFA setup response with secret and QR code."""

    secret: str = Field(..., description="TOTP secret key") # nosec B105 B106
    qr_code_uri: str = Field(..., description="URI for QR code generation")
    backup_codes: list[str] = Field(..., description="Backup recovery codes")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "secret": "JBSWY3DPEHPK3PXP", # nosec B105 B106
                "qr_code_uri": (
                    "otpauth://totp/BSOPT:user@example.com?" "secret=JBSWY3DPEHPK3PXP&issuer=BSOPT" # nosec B105 B106
                ),
                "backup_codes": ["12345678", "23456789", "34567890"],
            }
        }
    )


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
        json_schema_extra={"example": {"code": "123456"}}
    )


class EmailVerificationRequest(BaseModel):
    """Email verification request."""

    token: str = Field(..., description="Verification token from email") # nosec B105 B106

    model_config = ConfigDict(
        json_schema_extra={"example": {"token": "abc123def456"}} # nosec B105 B106
    )