"""
Authentication Routes (Optimized for PG16 + Async)
"""

from datetime import UTC, datetime, timedelta

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.exceptions import AuthenticationException, ConflictException, ValidationException
from src.api.schemas.auth import (
    LoginRequest,
    MFAVerifyRequest,
    PasswordChangeRequest,
    PasswordResetConfirmRequest,
    PasswordResetRequest,
    RegisterRequest,
)
from src.config import settings
from src.database import get_async_db, set_user_context
from src.database.models import User
from src.security.auth import (
    auth_service,
    get_current_active_user,
    get_current_user,
)
from src.security.mfa import mfa_service
from src.security.password import password_service
from src.security.rate_limit import rate_limit
from src.api.responses import MsgspecJSONResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"], dependencies=[Depends(rate_limit)], default_response_class=MsgspecJSONResponse)


def _log_legacy_warning(route: str):
    logger.warning("legacy_auth_route_accessed", route=route, migration_target="auth-service")


@router.post("/register", response_model=dict, status_code=status.HTTP_201_CREATED, deprecated=True)
async def register(
    data: RegisterRequest,
    background_tasks: BackgroundTasks,
    response: Response,
    db: AsyncSession = Depends(get_async_db),
):
    """
    [LEGACY] Register a new user using God-Mode Native DB procedure.
    MIGRATION: Use /api/auth/sign-up in the auth-service (Node.js).
    """
    _log_legacy_warning("/register")
    response.headers["X-API-Status"] = "deprecated"
    
    # 1. Validate password strength app-side
    val = password_service.validate_password(data.password)
    if not val.is_valid:
        raise ValidationException(message="Weak password", details=val.errors)

    # 2. Hand off registration + hashing to Postgres (Native Bcrypt)
    try:
        result = await db.execute(
            text("SELECT register_user_native(:email, :password, :name)"),
            {"email": data.email, "password": data.password, "name": data.full_name},
        )
        user_id = result.scalar()
        await db.commit()
    except Exception as e:
        await db.rollback()
        if "already registered" in str(e):
            raise ConflictException(message="Email already registered")
        logger.error(f"registration_native_failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failure")

    # 3. Fetch for JWT generation
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one()

    tokens = auth_service.create_token_pair(str(user.id), user.email, str(user.tier))

    return {
        "data": {
            "id": str(user.id),
            "email": user.email,
            "access_token": tokens.access_token,
            "token_type": tokens.token_type,
        },
        "message": "User created in God-Mode (Legacy)",
    }


@router.post("/login", response_model=dict, deprecated=True)
async def login(
    request: Request, 
    data: LoginRequest, 
    response: Response,
    db: AsyncSession = Depends(get_async_db)
):
    """
    [LEGACY] Authenticate via Native DB procedure (High Performance).
    MIGRATION: Use /api/auth/login in the auth-service (Node.js).
    """
    _log_legacy_warning("/login")
    response.headers["X-API-Status"] = "deprecated"
    
    try:
        # 1. Native Authentication (Handles hashing and active check DB-side)
        result = await db.execute(
            text("SELECT * FROM authenticate_user_native(:email, :password)"),
            {"email": data.email, "password": data.password},
        )
        row = result.fetchone()

        if not row:
            raise AuthenticationException(message="Invalid email or password")

        # row mapping: (id, email, tier, is_active)
        user_id, email, tier, is_active = row

        # 2. Sync session context for RLS
        await set_user_context(db, str(user_id))
        await db.commit()

        tokens = auth_service.create_token_pair(str(user_id), email, str(tier))
        return {
            "data": {
                "access_token": tokens.access_token,
                "refresh_token": tokens.refresh_token,
                "token_type": tokens.token_type,
                "expires_in": tokens.expires_in,
                "user_id": str(user_id),
                "email": email,
                "tier": str(tier),
            },
            "message": "Login successful (Legacy Native)",
        }
    except AuthenticationException:
        raise
    except Exception as e:
        logger.error(f"login_native_failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Authentication system error")


@router.get("/me")
async def read_users_me(user=Depends(get_current_active_user)):
    return {"data": user}


@router.post("/logout", deprecated=True)
async def logout(request: Request, response: Response, user=Depends(get_current_user)):
    """
    [LEGACY] Logout user.
    MIGRATION: Use /api/auth/sign-out in the auth-service (Node.js).
    """
    _log_legacy_warning("/logout")
    response.headers["X-API-Status"] = "deprecated"
    
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        await auth_service.invalidate_token(token, request)
    return {"message": "Successfully logged out"}


@router.post("/mfa/setup", deprecated=True)
async def mfa_setup(
    response: Response,
    user: User = Depends(get_current_active_user), 
    db: AsyncSession = Depends(get_async_db)
):
    """
    [LEGACY] Initialize MFA setup for the user.
    MIGRATION: Use auth-service's two-factor plugin routes.
    """
    _log_legacy_warning("/mfa/setup")
    response.headers["X-API-Status"] = "deprecated"
    
    if not user.mfa_secret:
        # Generate new plaintext secret
        plain_secret = mfa_service.generate_secret()
        # Encrypt before saving to DB
        user.mfa_secret = mfa_service.encrypt_secret(plain_secret)
        await db.commit()
    else:
        # Decrypt existing secret for URI generation
        plain_secret = mfa_service.decrypt_secret(user.mfa_secret)
    
    uri = mfa_service.get_provisioning_uri(user.email, plain_secret)
    qr_code = mfa_service.generate_qr_code(uri)
    
    return {
        "data": {
            "secret": plain_secret,  # Return plaintext once for setup
            "qr_code": qr_code
        },
        "message": "MFA setup initialized"
    }


@router.post("/mfa/verify", deprecated=True)
async def mfa_verify(
    data: MFAVerifyRequest, 
    response: Response,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    [LEGACY] Verify MFA code and enable it for the user.
    MIGRATION: Use auth-service's two-factor plugin routes.
    """
    _log_legacy_warning("/mfa/verify")
    response.headers["X-API-Status"] = "deprecated"
    
    if not user.mfa_secret:
        raise HTTPException(status_code=400, detail="MFA not initialized")
    
    # Decrypt secret for verification
    plain_secret = mfa_service.decrypt_secret(user.mfa_secret)
    
    if mfa_service.verify_code(plain_secret, data.code):
        user.is_mfa_enabled = True
        await db.commit()
        return {"status": "success", "message": "MFA enabled successfully (Legacy)"}
    raise ValidationException(message="Invalid MFA code")


@router.post("/password/change", deprecated=True)
async def change_password(
    data: PasswordChangeRequest, 
    response: Response,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    [LEGACY] Change user password.
    MIGRATION: Use /api/auth/change-password in auth-service.
    """
    _log_legacy_warning("/password/change")
    response.headers["X-API-Status"] = "deprecated"
    
    # 1. Verify old password
    if not password_service.verify_password(data.old_password, user.hashed_password):
        raise AuthenticationException(message="Invalid current password")
    
    # 2. Validate new password strength
    val = password_service.validate_password(data.new_password, user.email)
    if not val.is_valid:
        raise ValidationException(message="Weak new password", details=val.errors)
    
    # 3. Hash and save
    user.hashed_password = password_service.hash_password(data.new_password)
    await db.commit()
    
    return {"status": "success", "message": "Password changed successfully (Legacy)"}


@router.post("/password/reset", deprecated=True)
async def request_password_reset(
    data: PasswordResetRequest, 
    background_tasks: BackgroundTasks,
    response: Response,
    db: AsyncSession = Depends(get_async_db)
):
    """
    [LEGACY] Request a password reset email.
    MIGRATION: Use /api/auth/forget-password in auth-service.
    """
    _log_legacy_warning("/password/reset")
    response.headers["X-API-Status"] = "deprecated"
    
    result = await db.execute(select(User).where(User.email == data.email))
    user = result.scalar_one_or_none()
    
    if user:
        token = password_service.generate_reset_token()
        user.reset_token = token
        user.reset_token_expires_at = datetime.now(UTC) + timedelta(hours=1)
        await db.commit()
        
        background_tasks.add_task(_send_password_reset_email, user.email, token)
    
    # Always return success to prevent email enumeration
    return {"status": "success", "message": "If the email is registered, a reset link has been sent."}


@router.post("/password/reset/confirm", deprecated=True)
async def reset_password_confirm(
    data: PasswordResetConfirmRequest, 
    response: Response,
    db: AsyncSession = Depends(get_async_db)
):
    """
    [LEGACY] Confirm password reset with token.
    MIGRATION: Use /api/auth/reset-password in auth-service.
    """
    _log_legacy_warning("/password/reset/confirm")
    response.headers["X-API-Status"] = "deprecated"
    
    result = await db.execute(
        select(User).where(
            User.reset_token == data.token,
            User.reset_token_expires_at > datetime.now(UTC)
        )
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
    
    # Validate and hash new password
    val = password_service.validate_password(data.new_password, user.email)
    if not val.is_valid:
        raise ValidationException(message="Weak password", details=val.errors)
    
    user.hashed_password = password_service.hash_password(data.new_password)
    user.reset_token = None
    user.reset_token_expires_at = None
    await db.commit()
    
    return {"status": "success", "message": "Password has been reset successfully (Legacy)"}


# ---------------------------------------------------------------------------
# Internal Helpers (Mocked/Stubs for test compatibility)
# ---------------------------------------------------------------------------


async def _send_verification_email(email: str, token: str):
    """
    Sends a verification email to the user.
    In a real system, this would use an email service (e.g. SendGrid, Mailgun).
    """
    verification_link = f"{settings.BETTER_AUTH_URL}/verify-email?token={token}"
    logger.info("email_verification_link_generated", email=email, link=verification_link)
    # TODO: Integrate with actual email client (e.g. from src.utils.email)


async def _send_password_reset_email(email: str, token: str):
    """
    Sends a password reset email to the user.
    """
    reset_link = f"{settings.BETTER_AUTH_URL}/reset-password?token={token}"
    logger.info("password_reset_link_generated", email=email, link=reset_link)
    # TODO: Integrate with actual email client


def _verify_mfa_code(secret: str, code: str) -> bool:
    """Internal helper for MFA verification logic (legacy support)."""
    return mfa_service.verify_code(secret, code)
