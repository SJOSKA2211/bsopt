"""
Authentication Routes
=====================

Endpoints for user authentication:
- Login/logout
- Registration
- Token refresh
- Password reset
- MFA setup and verification
"""

import logging
import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from src.api.schemas.auth import (
    EmailVerificationRequest,
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
from src.api.schemas.common import ErrorResponse, SuccessResponse
from src.database import get_db
from src.database.models import User
from src.security.audit import AuditEvent, log_audit
from src.security.auth import (
    auth_service,
    get_current_active_user,
    get_current_user,
)
from src.security.password import password_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# =============================================================================
# Login / Logout
# =============================================================================


@router.post(
    "/login",
    response_model=LoginResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid credentials"},
        403: {"model": ErrorResponse, "description": "Account disabled or unverified"},
    },
)
async def login(
    request: Request,
    login_data: LoginRequest,
    db: Session = Depends(get_db),
):
    """
    Authenticate user and return JWT tokens.

    - **email**: User's email address
    - **password**: User's password
    - **remember_me**: If true, extends token expiration
    - **mfa_code**: Required if MFA is enabled
    """
    # Authenticate user
    user = auth_service.authenticate_user(
        db=db,
        email=login_data.email,
        password=login_data.password,
        request=request,
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if email is verified
    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified. Please check your email for verification link.",
        )

    # Check MFA if enabled
    if user.is_mfa_enabled:
        if not login_data.mfa_code:
            return LoginResponse(
                access_token="",
                refresh_token="",
                expires_in=0,
                user_id=str(user.id),
                email=user.email,
                tier=user.tier,
                requires_mfa=True,
            )

        # Verify MFA code
        if not _verify_mfa_code(user, login_data.mfa_code):
            log_audit(AuditEvent.MFA_LOGIN_FAILURE, user=user, request=request)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid MFA code",
            )
        log_audit(AuditEvent.MFA_LOGIN_SUCCESS, user=user, request=request)

    # Update last login
    user.last_login = datetime.now(timezone.utc)
    db.commit()

    # Create tokens
    token_pair = auth_service.create_token_pair(
        user_id=str(user.id),
        email=user.email,
        tier=user.tier,
    )

    return LoginResponse(
        access_token=token_pair.access_token,
        refresh_token=token_pair.refresh_token,
        token_type=token_pair.token_type,
        expires_in=token_pair.expires_in,
        user_id=str(user.id),
        email=user.email,
        tier=user.tier,
        requires_mfa=False,
    )


@router.post("/logout", response_model=SuccessResponse)
async def logout(
    request: Request,
    token: str = Depends(auth_service.validate_token),
    user: User = Depends(get_current_user),
):
    """
    Logout user by invalidating their tokens.
    """
    # Get token from header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        await auth_service.invalidate_token(token, request)

    log_audit(AuditEvent.USER_LOGOUT, user=user, request=request)

    return SuccessResponse(message="Successfully logged out")


# =============================================================================
# Registration
# =============================================================================


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        409: {"model": ErrorResponse, "description": "Email already registered"},
    },
)
async def register(
    request: Request,
    register_data: RegisterRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Register a new user account.

    - Validates password strength
    - Sends verification email
    - Returns user ID on success
    """
    # Check if email already exists
    existing_user = db.query(User).filter(User.email == register_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )

    # Validate password
    validation = password_service.validate_password(register_data.password, register_data.email)
    if not validation.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="; ".join(validation.errors),
        )

    # Create user
    verification_token = secrets.token_urlsafe(32)

    user = User(
        email=register_data.email,
        hashed_password=password_service.hash_password(register_data.password),
        full_name=register_data.full_name,
        tier="free",
        is_active=True,
        is_verified=False,
        verification_token=verification_token,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    # Log registration
    log_audit(AuditEvent.USER_REGISTER, user=user, request=request)

    # Queue verification email
    background_tasks.add_task(
        _send_verification_email,
        user.email,
        verification_token,
    )

    return RegisterResponse(
        user_id=str(user.id),
        email=user.email,
        message="Registration successful. Please check your email to verify your account.",
        verification_required=True,
    )


@router.post("/verify-email", response_model=SuccessResponse)
async def verify_email(
    request: Request,
    verification: EmailVerificationRequest,
    db: Session = Depends(get_db),
):
    """
    Verify user email with token from email link.
    """
    user = db.query(User).filter(User.verification_token == verification.token).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token",
        )

    user.is_verified = True
    user.verification_token = None
    db.commit()

    return SuccessResponse(message="Email verified successfully. You can now log in.")


# =============================================================================
# Token Refresh
# =============================================================================


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: Request,
    refresh_data: RefreshTokenRequest,
    db: Session = Depends(get_db),
):
    """
    Refresh access token using refresh token.
    """
    try:
        token_data = await auth_service.validate_token(refresh_data.refresh_token)

        if token_data.token_type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )

        # Get user
        user = db.query(User).filter(User.id == token_data.user_id).first()
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive",
            )

        # Invalidate old refresh token
        await auth_service.invalidate_token(refresh_data.refresh_token)

        # Create new token pair
        token_pair = auth_service.create_token_pair(
            user_id=str(user.id),
            email=user.email,
            tier=user.tier,
        )

        log_audit(AuditEvent.TOKEN_REFRESH, user=user, request=request)

        return TokenResponse(
            access_token=token_pair.access_token,
            refresh_token=token_pair.refresh_token,
            token_type=token_pair.token_type,
            expires_in=token_pair.expires_in,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not refresh token",
        )


# =============================================================================
# Password Reset
# =============================================================================


@router.post("/password-reset", response_model=SuccessResponse)
async def request_password_reset(
    request: Request,
    reset_data: PasswordResetRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Request password reset email.

    Always returns success to prevent email enumeration.
    """
    user = db.query(User).filter(User.email == reset_data.email).first()

    if user:
        # Generate reset token
        reset_token = secrets.token_urlsafe(32)

        # Store token (in production, store in Redis with TTL)
        # For now, we'll use a simple approach
        user.verification_token = f"reset:{reset_token}"
        db.commit()

        log_audit(AuditEvent.PASSWORD_RESET_REQUEST, user=user, request=request)

        # Queue reset email
        background_tasks.add_task(
            _send_password_reset_email,
            user.email,
            reset_token,
        )

    # Always return success to prevent email enumeration
    return SuccessResponse(
        message="If an account with this email exists, a password reset link has been sent."
    )


@router.post("/password-reset/confirm", response_model=SuccessResponse)
async def confirm_password_reset(
    request: Request,
    reset_data: PasswordResetConfirm,
    db: Session = Depends(get_db),
):
    """
    Confirm password reset with token and new password.
    """
    # Find user by reset token
    user = db.query(User).filter(User.verification_token == f"reset:{reset_data.token}").first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )

    # Validate new password
    validation = password_service.validate_password(reset_data.new_password, user.email)
    if not validation.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="; ".join(validation.errors),
        )

    # Update password
    user.hashed_password = password_service.hash_password(reset_data.new_password)
    user.verification_token = None
    db.commit()

    log_audit(AuditEvent.PASSWORD_RESET_SUCCESS, user=user, request=request)

    return SuccessResponse(
        message="Password has been reset successfully. You can now log in with your new password."
    )


@router.post("/password-change", response_model=SuccessResponse)
async def change_password(
    request: Request,
    change_data: PasswordChangeRequest,
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Change password for authenticated user.
    """
    # Verify current password
    if not password_service.verify_password(change_data.current_password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect",
        )

    # Validate new password
    validation = password_service.validate_password(change_data.new_password, user.email)
    if not validation.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="; ".join(validation.errors),
        )

    # Update password
    user.hashed_password = password_service.hash_password(change_data.new_password)
    db.commit()

    log_audit(AuditEvent.PASSWORD_CHANGED, user=user, request=request)

    return SuccessResponse(message="Password changed successfully")


# =============================================================================
# MFA
# =============================================================================


@router.post("/mfa/setup", response_model=MFASetupResponse)
async def setup_mfa(
    request: Request,
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Setup MFA for the authenticated user.
    Returns secret and QR code URI.
    """
    import pyotp

    # Generate secret
    secret = pyotp.random_base32()

    # Generate QR code URI
    totp = pyotp.TOTP(secret)
    qr_uri = totp.provisioning_uri(name=user.email, issuer_name="BSOPT")

    # Generate backup codes
    backup_codes = [secrets.token_hex(4) for _ in range(8)]

    # Store secret temporarily (user must verify before enabling)
    user.mfa_secret = secret
    db.commit()

    return MFASetupResponse(
        secret=secret,
        qr_code_uri=qr_uri,
        backup_codes=backup_codes,
    )


@router.post("/mfa/verify", response_model=SuccessResponse)
async def verify_mfa(
    request: Request,
    verify_data: MFAVerifyRequest,
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Verify MFA code to enable MFA.
    """
    if not user.mfa_secret:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MFA setup not initiated",
        )

    if not _verify_mfa_code(user, verify_data.code):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid MFA code",
        )

    # Enable MFA
    user.is_mfa_enabled = True
    db.commit()

    log_audit(AuditEvent.MFA_ENABLED, user=user, request=request)

    return SuccessResponse(message="MFA enabled successfully")


@router.post("/mfa/disable", response_model=SuccessResponse)
async def disable_mfa(
    request: Request,
    verify_data: MFAVerifyRequest,
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Disable MFA for the authenticated user.
    Requires current MFA code for security.
    """
    if not user.is_mfa_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MFA is not enabled",
        )

    if not _verify_mfa_code(user, verify_data.code):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid MFA code",
        )

    # Disable MFA
    user.is_mfa_enabled = False
    user.mfa_secret = None
    db.commit()

    log_audit(AuditEvent.MFA_DISABLED, user=user, request=request)

    return SuccessResponse(message="MFA disabled successfully")


# =============================================================================
# Helper Functions
# =============================================================================


def _verify_mfa_code(user: User, code: str) -> bool:
    """Verify TOTP code."""
    import pyotp

    if not user.mfa_secret:
        return False

    totp = pyotp.TOTP(user.mfa_secret)
    return totp.verify(code, valid_window=1)


async def _send_verification_email(email: str, token: str) -> None:
    """Send verification email via Celery."""
    from src.tasks.email_tasks import send_transactional_email

    send_transactional_email.delay(
        to_email=email,
        subject="Verify your BSOPT account",
        template_name="verification.html",
        context={"token": token},
    )


async def _send_password_reset_email(email: str, token: str) -> None:
    """Send password reset email via Celery."""
    from src.tasks.email_tasks import send_transactional_email

    send_transactional_email.delay(
        to_email=email,
        subject="Reset your BSOPT password",
        template_name="password_reset.html",
        context={"token": token},
    )
