import pyotp
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
import hashlib
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Request, status, HTTPException
from sqlalchemy.orm import Session
from cryptography.fernet import Fernet

from src.config import settings
from src.api.exceptions import (
    AuthenticationException,
    ConflictException,
    InternalServerException,
    PermissionDeniedException,
    ValidationException,
)
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
from src.api.schemas.common import DataResponse, ErrorResponse, SuccessResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from anyio.to_thread import run_sync
from src.database import get_async_db, get_db
from src.database.models import User
from src.security.auth import get_auth_service, TokenData, get_token_from_header
from src.security.password import get_password_service
from src.utils.sanitization import sanitize_string
from src.utils.cache import idempotency_manager
from src.tasks.email_tasks import send_transactional_email
from src.security.audit import AuditEvent, log_audit

logger = logging.getLogger(__name__)


async def get_current_user(
    request: Request,
    token: Optional[str] = Depends(get_token_from_header),
    db: AsyncSession = Depends(get_async_db),
) -> User:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data: TokenData = await get_auth_service().validate_token(token)
    user_id = token_data.user_id

    try:
        from src.utils.cache import db_cache
        cached_user_data = await db_cache.get_user(user_id)
        if cached_user_data:
            user = User(**cached_user_data)
            request.state.user = user
            return user
    except Exception as e:
        logger.warning(f"Cache lookup failed: {e}")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        from src.utils.cache import db_cache
        user_data_dict = {c.name: getattr(user, c.name) for c in user.__table__.columns}
        await db_cache.set_user(user_id, user_data_dict)
    except Exception as e:
        logger.warning(f"Failed to set user in cache: {e}")

    request.state.user = user
    return user


async def get_current_active_user(
    user: User = Depends(get_current_user),
) -> User:
    """Get current user and verify they are active."""
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )
    return user

router = APIRouter(prefix="/auth", tags=["Authentication"])


# =============================================================================
# Login / Logout
# =============================================================================


@router.post(
    "/login",
    response_model=DataResponse[LoginResponse],
)
async def login(
    request: Request,
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_async_db),
    auth_service = Depends(get_auth_service),
):
    """Authenticate user (Async)."""
    # Authenticate user
    user = await auth_service.authenticate_user(db, login_data.email, login_data.password, request)

    if not user:
        raise AuthenticationException(message="Invalid email or password")

    if not user.is_verified:
        raise PermissionDeniedException(message="Email not verified")

    if user.is_mfa_enabled:
        if not login_data.mfa_code:
            return DataResponse(
                data=LoginResponse(
                    access_token="", refresh_token="", expires_in=0,
                    user_id=str(user.id), email=user.email, tier=user.tier,
                    requires_mfa=True
                ),
                message="MFA required"
            )

        if not await run_sync(_verify_mfa_code, user, login_data.mfa_code, db):
            raise AuthenticationException(message="Invalid MFA code")

    # Update last login
    try:
        user.last_login = datetime.now(timezone.utc)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to update last login for user {user.id}: {e}")

    token_pair = auth_service.create_token_pair(str(user.id), user.email, user.tier)

    return DataResponse(
        data=LoginResponse(
            access_token=token_pair.access_token,
            refresh_token=token_pair.refresh_token,
            token_type=token_pair.token_type,
            expires_in=token_pair.expires_in,
            user_id=str(user.id),
            email=user.email,
            tier=user.tier,
            requires_mfa=False
        )
    )



@router.post(
    "/logout",
    response_model=SuccessResponse,
    responses={401: {"model": ErrorResponse}},
)
async def logout(
    request: Request,
    user: User = Depends(get_current_user),
    auth_service = Depends(get_auth_service),
):
    """
    Invalidate the current session's tokens and log the user out.
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        await auth_service.invalidate_token(token, request)

    log_audit(AuditEvent.USER_LOGOUT, user=user, request=request)

    return SuccessResponse(message="Successfully logged out and session invalidated")


# =============================================================================
# Registration
# =============================================================================


@router.post(
    "/register",
    response_model=DataResponse[RegisterResponse],
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Validation/Password error"},
        409: {"model": ErrorResponse, "description": "Email already registered"},
        422: {"model": ErrorResponse, "description": "Schema validation error"},
    },
)
async def register(
    request: Request,
    register_data: RegisterRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_db),
    password_service = Depends(get_password_service),
):
    """
    Create a new user account.
    """
    # ... (idempotency check)
    
    # Check if email already exists
    result = await db.execute(select(User).where(User.email == register_data.email))
    existing_user = result.scalar_one_or_none()
    if existing_user:
        raise ConflictException(
            message="An account with this email address already exists",
        )

    # Validate password
    validation = password_service.validate_password(register_data.password, register_data.email)
    if not validation.is_valid:
        raise ValidationException(
            message=f"Password policy violation: {'; '.join(validation.errors)}",
        )

    # Create user
    verification_token = password_service.generate_verification_token()

    try:
        user = User(
            email=register_data.email,
            hashed_password=await run_sync(password_service.hash_password, register_data.password),
            full_name=sanitize_string(register_data.full_name) if register_data.full_name else None,
            tier="free",
            is_active=True,
            is_verified=False,
            verification_token=f"verify:{verification_token}",
        )

        db.add(user)
        await db.commit()
        await db.refresh(user)
    except Exception as e:
        await db.rollback()
        logger.error(f"Error during user registration: {e}")
        raise InternalServerException(
            message="Failed to create user account",
        )

    # Log registration
    log_audit(AuditEvent.USER_REGISTER, user=user, request=request)

    # Queue verification email
    background_tasks.add_task(
        _send_verification_email,
        user.email,
        verification_token,
    )

    return DataResponse(
        data=RegisterResponse(
            user_id=str(user.id),
            email=user.email,
            message="Registration successful. A verification link has been sent to your email.",
            verification_required=True,
        ),
        message="User registered successfully"
    )


@router.post(
    "/verify-email",
    response_model=SuccessResponse,
    responses={400: {"model": ErrorResponse}},
)
async def verify_email(
    request: Request,
    verification: EmailVerificationRequest,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Complete the registration process by verifying the user's email address using a token.
    """
    result = await db.execute(select(User).where(User.verification_token == f"verify:{verification.token}"))
    user = result.scalar_one_or_none()
    if not user:
        raise ValidationException(
            message="The verification token is invalid or has already been used",
        )

    try:
        user.is_verified = True
        user.verification_token = None
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to update verification status for user {user.id}: {e}")
        raise InternalServerException(
            message="Internal error during email verification",
        )

    return SuccessResponse(message="Email verified successfully. You may now log in to your account.")



# =============================================================================
# Token Refresh
# =============================================================================


@router.post(
    "/refresh",
    response_model=DataResponse[TokenResponse],
    responses={401: {"model": ErrorResponse}},
)
async def refresh_token(
    request: Request,
    refresh_data: RefreshTokenRequest,
    db: AsyncSession = Depends(get_async_db),
    auth_service = Depends(get_auth_service),
):
    """
    Get a new access token using a refresh token.
    """
    try:
        token_data = await auth_service.validate_token(refresh_data.refresh_token)
        if token_data.token_type != "refresh":
            raise AuthenticationException(message="Invalid token type")

        result = await db.execute(select(User).where(User.id == token_data.user_id))
        user = result.scalar_one_or_none()
        if not user or not user.is_active:
            raise AuthenticationException(message="User not found or inactive")

        await auth_service.invalidate_token(refresh_data.refresh_token, request)

        token_pair = auth_service.create_token_pair(
            user_id=str(user.id),
            email=user.email,
            tier=user.tier,
        )

        log_audit(AuditEvent.TOKEN_REFRESH, user=user, request=request)

        return DataResponse(
            data=TokenResponse(
                access_token=token_pair.access_token,
                refresh_token=token_pair.refresh_token,
                token_type=token_pair.token_type,
                expires_in=token_pair.expires_in,
            ),
            message="Token refreshed successfully"
        )

    except (AuthenticationException, PermissionDeniedException):
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise AuthenticationException(
            message="Session could not be refreshed. Please log in again.",
        )


# =============================================================================
# Password Reset
# =============================================================================


@router.post(
    "/password-reset",
    response_model=SuccessResponse,
    responses={422: {"model": ErrorResponse}},
)
async def request_password_reset(
    request: Request,
    reset_data: PasswordResetRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Initiate the password recovery process.
    """
    result = await db.execute(select(User).where(User.email == reset_data.email))
    user = result.scalar_one_or_none()

    if user:
        reset_token = secrets.token_urlsafe(32)

        try:
            user.verification_token = f"reset:{reset_token}"
            await db.commit()
        except Exception as e:
            await db.rollback()
            logger.error(f"Error saving reset token for user {user.id}: {e}")

        log_audit(AuditEvent.PASSWORD_RESET_REQUEST, user=user, request=request)

        background_tasks.add_task(
            _send_password_reset_email,
            user.email,
            reset_token,
        )
    else:
        # Timing attack mitigation: Simulate work to match the 'user found' path
        # 1. Generate a dummy token
        _ = secrets.token_urlsafe(32)
        
        # 2. Simulate a DB write/logic delay
        # We perform a dummy hash which is computationally expensive similar to the token prefix logic
        _ = hashlib.sha256(b"dummy_timing_mitigation_salt").hexdigest()
        
        # 3. Queue a dummy email task (or just sleep briefly)
        # Note: In production, we might want to actually queue a 'no-op' task to the same queue
        # to ensure queue latency is also matched.
        background_tasks.add_task(
            _send_password_reset_email,
            "dummy@example.com", # Use a dummy email to avoid sending actual emails
            _ # Use the dummy token generated earlier
        )

    return SuccessResponse(
        message="If an account with this email exists, a password reset link has been sent."
    )


@router.post(
    "/password-reset/confirm",
    response_model=SuccessResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def confirm_password_reset(
            request: Request,
            reset_data: PasswordResetConfirm,
            db: AsyncSession = Depends(get_async_db),
        ):
            """
            Finalize password recovery using the token received via email.
            """
            result = await db.execute(select(User).where(User.verification_token == f"reset:{reset_data.token}"))
            user = result.scalar_one_or_none()
        
            if not user:
                raise ValidationException(
                    message="The reset token provided is invalid or has expired",
                )
        
            validation = get_password_service().validate_password(reset_data.new_password, user.email)
            if not validation.is_valid:
                raise ValidationException(
                    message=f"New password is not secure enough: {'; '.join(validation.errors)}",
                )
        
            try:
                user.hashed_password = await run_sync(get_password_service().hash_password, reset_data.new_password)
                user.verification_token = None
                await db.commit()
            except Exception as e:
                await db.rollback()
                logger.error(f"Error updating password after reset for user {user.id}: {e}")
                raise InternalServerException(
                    message="Failed to update account password",
                )
        
            log_audit(AuditEvent.PASSWORD_RESET_SUCCESS, user=user, request=request)
        
            return SuccessResponse(
                message="Your password has been successfully reset. You may now log in with your new credentials."
            )

@router.post(
    "/password-change",
    response_model=SuccessResponse,
    responses={401: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
async def change_password(
    request: Request,
    change_data: PasswordChangeRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Update the password for the currently logged-in user.
    """
    if not await run_sync(get_password_service().verify_password, change_data.current_password, user.hashed_password):
        raise AuthenticationException(
            message="The current password provided is incorrect",
        )

    validation = get_password_service().validate_password(change_data.new_password, user.email)
    if not validation.is_valid:
        raise ValidationException(
            message=f"New password does not meet requirements: {'; '.join(validation.errors)}",
        )

    try:
        user.hashed_password = await run_sync(get_password_service().hash_password, change_data.new_password)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Error changing password for user {user.id}: {e}")
        raise InternalServerException(
            message="Internal error updating password",
        )

    log_audit(AuditEvent.PASSWORD_CHANGED, user=user, request=request)

    return SuccessResponse(message="Your password has been changed successfully")


# =============================================================================
# MFA
# =============================================================================


@router.post(
    "/mfa/setup",
    response_model=DataResponse[MFASetupResponse],
    responses={401: {"model": ErrorResponse}},
)
async def setup_mfa(
    request: Request,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Begin Multi-Factor Authentication setup. 
    """
    secret = pyotp.random_base32()
    totp = pyotp.TOTP(secret)
    qr_uri = totp.provisioning_uri(name=user.email, issuer_name="BSOPT")

    backup_codes = [secrets.token_hex(4) for _ in range(8)]
    hashed_backup_codes = [hashlib.sha256(bc.encode()).hexdigest() for bc in backup_codes]

    try:
        fernet = Fernet(settings.MFA_ENCRYPTION_KEY)
        encrypted_secret = fernet.encrypt(secret.encode())
        user.mfa_secret = encrypted_secret.decode()
        user.mfa_backup_codes = ",".join(hashed_backup_codes)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Error saving MFA secret for user {user.id}: {e}")
        raise InternalServerException(
            message="Failed to initiate MFA setup",
        )

    return DataResponse(
        data=MFASetupResponse(
            secret=secret,
            qr_code_uri=qr_uri,
            backup_codes=backup_codes,
        ),
        message="MFA setup initiated"
    )


@router.post(
    "/mfa/verify",
    response_model=SuccessResponse,
    responses={401: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
async def verify_mfa(
    request: Request,
    verify_data: MFAVerifyRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Verify the MFA code to complete activation.
    """
    if not user.mfa_secret:
        raise ValidationException(
            message="MFA setup has not been initiated for this account",
        )

    if not await run_sync(_verify_mfa_code, user, verify_data.code, db):
        raise AuthenticationException(
            message="The Multi-Factor Authentication code provided is invalid",
        )

    try:
        user.is_mfa_enabled = True
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Error activating MFA for user {user.id}: {e}")
        raise InternalServerException(
            message="Internal error enabling Multi-Factor Authentication",
        )

    log_audit(AuditEvent.MFA_ENABLED, user=user, request=request)

    return SuccessResponse(message="Multi-Factor Authentication has been enabled successfully")


@router.post(
    "/mfa/disable",
    response_model=SuccessResponse,
    responses={401: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
async def disable_mfa(
    request: Request,
    verify_data: MFAVerifyRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Deactivate Multi-Factor Authentication. 
    """
    if not user.is_mfa_enabled:
        raise ValidationException(
            message="Multi-Factor Authentication is already disabled for this account",
        )

    if not await run_sync(_verify_mfa_code, user, verify_data.code, db):
        raise AuthenticationException(
            message="Invalid confirmation code provided",
        )

    try:
        user.is_mfa_enabled = False
        user.mfa_secret = None
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Error disabling MFA for user {user.id}: {e}")
        raise InternalServerException(
            message="Internal error deactivating Multi-Factor Authentication",
        )

    log_audit(AuditEvent.MFA_DISABLED, user=user, request=request)

    return SuccessResponse(message="Multi-Factor Authentication has been disabled successfully")


# =============================================================================
# Helper Functions
# =============================================================================


def _verify_mfa_code(user: User, code: str, db: Optional[Session] = None) -> bool:
    """Verify TOTP code or backup code."""
    if not user.mfa_secret:
        return False

    # 1. Try TOTP verification
    try:
        fernet = Fernet(settings.MFA_ENCRYPTION_KEY)
        decrypted_secret = fernet.decrypt(user.mfa_secret.encode()).decode()
        totp = pyotp.TOTP(decrypted_secret)
        if totp.verify(code, valid_window=1):
            return True
    except Exception:
        pass

    # 2. Try backup code verification if TOTP failed
    if user.mfa_backup_codes:
        hashed_code = hashlib.sha256(code.encode()).hexdigest()
        backup_list = user.mfa_backup_codes.split(",")
        if hashed_code in backup_list:
            if db:
                # Remove used backup code
                backup_list.remove(hashed_code)
                user.mfa_backup_codes = ",".join(backup_list)
                # Note: Caller is responsible for final commit
            return True

    return False



async def _send_verification_email(email: str, token: str) -> None:
    """Send verification email via Celery."""
    send_transactional_email.delay(
        to_email=email,
        subject="Verify your BSOPT account",
        template_name="verification.html",
        context={"token": token},
    )


async def _send_password_reset_email(email: str, token: str) -> None:
    """Send password reset email via Celery."""
    send_transactional_email.delay(
        to_email=email,
        subject="Reset your BSOPT password",
        template_name="password_reset.html",
        context={"token": token},
    )