"""
Authentication Routes (Optimized for PG16 + Async)
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status
from jwt.exceptions import PyJWTError
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.exceptions import AuthenticationException, ConflictException, ValidationException
from services.api.responses import MsgspecJSONResponse
from services.api.schemas.auth import (
    LoginRequest,
    LoginResponse,
    MFASetupResponse,
    MFAVerifyRequest,
    PasswordChangeRequest,
    PasswordResetConfirmRequest,
    PasswordResetRequest,
    RefreshTokenRequest,
    RegisterRequest,
    TokenResponse,
)
from services.api.schemas.common import DataResponseStruct, SuccessResponse
from services.api.schemas.user import UserResponse
from src.auth.auth import (
    auth_service,
    get_current_active_user,
    get_current_user,
)
from src.auth.mfa import mfa_service
from src.auth.password import password_service
from src.auth.rate_limit import rate_limit
from src.config import settings
from src.database import get_async_db, set_user_context
from src.database.models import User

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
    dependencies=[Depends(rate_limit)],
    default_response_class=MsgspecJSONResponse,
)


@router.post("/register", response_model=DataResponseStruct[TokenResponse], status_code=status.HTTP_201_CREATED)
async def register(
    data: RegisterRequest,
    background_tasks: BackgroundTasks,
    response: Response,
    db: AsyncSession = Depends(get_async_db),
) -> DataResponseStruct[TokenResponse]:
    """
    Register a new user using High-Performance Native DB procedure.
    """

    # 1. Hashing and registration are offloaded to Postgres (Native Bcrypt)
    # Validation is handled by Pydantic (RegisterRequest)

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

    return DataResponseStruct(
        data=TokenResponse(
            user_id=str(user.id),
            email=user.email,
            access_token=tokens.access_token,
            token_type=tokens.token_type,
        ),
        message="User created in High-Performance (Legacy)",
    )


@router.post("/login", response_model=DataResponseStruct[LoginResponse])
async def login(
    request: Request,
    data: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_async_db),
) -> DataResponseStruct[LoginResponse]:
    """
    Authenticate via Native DB procedure (High Performance).
    """

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
        user_id, email, tier, _ = row

        # 2. Sync session context for RLS
        await set_user_context(db, str(user_id))
        await db.commit()

        tokens = auth_service.create_token_pair(str(user_id), email, str(tier))
        return DataResponseStruct(
            data=LoginResponse(
                access_token=tokens.access_token,
                refresh_token=tokens.refresh_token,
                token_type=tokens.token_type,
                expires_in=tokens.expires_in,
                user_id=str(user_id),
                email=email,
                tier=str(tier),
            ),
            message="Login successful",
        )
    except AuthenticationException:
        raise
    except Exception as e:
        logger.error(f"login_failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Authentication failure")


@router.post("/refresh", response_model=DataResponseStruct[TokenResponse])
async def refresh_token(
    data: RefreshTokenRequest,
    db: AsyncSession = Depends(get_async_db),
) -> DataResponseStruct[TokenResponse]:
    """
    Refresh access token using a valid refresh token.
    Implements Refresh Token Rotation for enhanced security.
    """
    try:
        # 1. Decode and validate the refresh token
        token_data = auth_service.decode_token(data.refresh_token)
        if token_data.token_type != "refresh":
            raise AuthenticationException(message="Invalid token type")

        # 2. Check blacklist (Reuse detection)
        if await auth_service.token_blacklist.contains(token_data.jti):
            logger.warning(
                "refresh_token_reuse_detected", jti=token_data.jti, user_id=token_data.user_id
            )
            # Potentially revoke all tokens for this user for safety
            raise AuthenticationException(message="Token has been revoked")

        # 3. Invalidate the used refresh token (Rotation)
        await auth_service.token_blacklist.add(token_data.jti, token_data.exp)

        # 4. Create new token pair
        new_tokens = auth_service.create_token_pair(
            token_data.user_id, token_data.email, token_data.tier
        )

        return DataResponseStruct(
            data=TokenResponse(
                access_token=new_tokens.access_token,
                refresh_token=new_tokens.refresh_token,
                token_type=new_tokens.token_type,
                expires_in=new_tokens.expires_in,
                user_id=token_data.user_id,
                email=token_data.email,
                tier=token_data.tier,
            ),
            message="Token refreshed successfully",
        )
    except PyJWTError:
        raise AuthenticationException(message="Invalid or expired refresh token")
    except Exception as e:
        logger.error(f"refresh_failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Token refresh failure")


@router.get("/me")
async def read_users_me(user: User = Depends(get_current_active_user)):
    return DataResponseStruct(data=UserResponse.from_orm(user))


@router.post("/logout")
async def logout(
    request: Request, response: Response, user: User = Depends(get_current_user)
) -> SuccessResponse:
    """
    Logout user.
    """

    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        await auth_service.invalidate_token(token, request)
    return SuccessResponse(message="Successfully logged out")


@router.post("/mfa/setup", response_model=DataResponseStruct[MFASetupResponse])
async def mfa_setup(
    response: Response,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
) -> DataResponseStruct[MFASetupResponse]:
    """
    Initialize MFA setup for the user.
    """

    if not user.mfa_secret:
        # Generate new plaintext secret
        plain_secret = mfa_service.generate_secret()
        # Encrypt before saving to DB
        user.mfa_secret = mfa_service.encrypt_secret(plain_secret)
        await db.commit()
    else:
        # Decrypt existing secret for URI generation
        plain_secret = mfa_service.decrypt_secret(user.mfa_secret)

    # Generate provisioning URI
    uri = mfa_service.generate_provisioning_uri(user.email, plain_secret)

    return DataResponseStruct(
        data=MFASetupResponse(
            secret=plain_secret,
            provisioning_uri=uri,
            qr_code_uri=None,  # Frontend generates QR from URI
            backup_codes=[],  # Future: generate and return backup codes
        )
    )


@router.post("/mfa/verify")
async def mfa_verify(
    data: MFAVerifyRequest,
    response: Response,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
) -> SuccessResponse:
    """
    Verify MFA code and enable MFA for the user.
    """

    if not user.mfa_secret:
        raise HTTPException(status_code=400, detail="MFA not setup")

    # Decrypt secret
    plain_secret = mfa_service.decrypt_secret(user.mfa_secret)

    if not mfa_service.verify_code(plain_secret, data.code):
        raise AuthenticationException(message="Invalid MFA code")

    # Enable MFA
    user.mfa_enabled = True
    await db.commit()

    return SuccessResponse(message="MFA enabled successfully")


@router.post("/password/change")
async def change_password(
    data: PasswordChangeRequest,
    response: Response,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
) -> SuccessResponse:
    """
    Change user password.
    """

    # 1. Verify old password
    if not password_service.verify_password(data.current_password, user.hashed_password):
        raise AuthenticationException(message="Invalid current password")

    # 2. Update password
    # Validation is handled by Pydantic (PasswordChangeRequest)

    # 3. Hash and save
    user.hashed_password = password_service.hash_password(data.new_password)
    await db.commit()

    return SuccessResponse(message="Password changed successfully")


@router.post("/password/reset")
async def request_password_reset(
    data: PasswordResetRequest,
    background_tasks: BackgroundTasks,
    response: Response,
    db: AsyncSession = Depends(get_async_db),
) -> SuccessResponse:
    """
    Request a password reset email.
    """

    result = await db.execute(select(User).where(User.email == data.email))
    user = result.scalar_one_or_none()

    if user:
        token = password_service.generate_reset_token()
        user.reset_token = token
        user.reset_token_expires_at = datetime.now(UTC) + timedelta(hours=1)
        await db.commit()

        background_tasks.add_task(_send_password_reset_email, user.email, token)

    # Always return success to prevent email enumeration
    return SuccessResponse(
        message="If the email is registered, a reset link has been sent.",
    )


@router.post("/password/reset/confirm")
async def reset_password_confirm(
    data: PasswordResetConfirmRequest, response: Response, db: AsyncSession = Depends(get_async_db)
) -> SuccessResponse:
    """
    Confirm password reset with token.
    """

    result = await db.execute(
        select(User).where(
            User.reset_token == data.token, User.reset_token_expires_at > datetime.now(UTC)
        )
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    # 1. Update password
    # Validation is handled by Pydantic (PasswordResetConfirmRequest)

    user.hashed_password = password_service.hash_password(data.new_password)
    user.reset_token = None
    user.reset_token_expires_at = None
    await db.commit()

    return SuccessResponse(message="Password has been reset successfully")


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


async def _send_verification_email(email: str, token: str) -> None:
    """
    Sends a verification email to the user using Celery.
    """
    verification_link = f"{settings.BETTER_AUTH_URL}/verify-email?token={token}"
    # Omit the full link from logs for security
    logger.info("email_verification_link_generated", email=email, link="[REDACTED]")

    from src.workers.tasks.email_tasks import send_transactional_email

    send_transactional_email.delay(
        to_email=email,
        subject="Verify Your BSOpt Account",
        template_name="verification_email.html",
        context={"verification_link": verification_link, "email": email},
    )


async def _send_password_reset_email(email: str, token: str) -> None:
    """
    Sends a password reset email to the user using Celery.
    """
    reset_link = f"{settings.BETTER_AUTH_URL}/reset-password?token={token}"
    # Omit the full link from logs for security
    logger.info("password_reset_link_generated", email=email, link="[REDACTED]")

    from src.workers.tasks.email_tasks import send_transactional_email

    send_transactional_email.delay(
        to_email=email,
        subject="Password Reset Request",
        template_name="password_reset.html",
        context={"reset_link": reset_link, "email": email},
    )
