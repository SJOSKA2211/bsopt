from datetime import UTC, datetime, timedelta

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status
from jwt.exceptions import PyJWTError
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from api.exceptions import AuthenticationException, ConflictException
from api.responses import MsgspecJSONResponse
from api.schemas.auth import (
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
    WebAuthnAuthenticationVerificationRequest,
    WebAuthnRegistrationVerificationRequest,
)
from api.schemas.common import DataResponse, SuccessResponse
from api.schemas.user import UserResponse
from src.auth.auth import (
    auth_service,
    get_current_active_user,
    get_current_user,
)
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


@router.post(
    "/register", response_model=DataResponse[TokenResponse], status_code=status.HTTP_201_CREATED
)
async def register(
    data: RegisterRequest,
    background_tasks: BackgroundTasks,
    response: Response,
    db: AsyncSession = Depends(get_async_db),
) -> DataResponse[TokenResponse]:
    """
    Register a new user.
    """
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

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one()

    tokens = auth_service.create_token_pair(str(user.id), user.email, str(user.tier))

    return DataResponse(
        data=TokenResponse(
            user_id=str(user.id),
            email=user.email,
            access_token=tokens.access_token,
            token_type=tokens.token_type,
        ),
        message="User created",
    )


@router.post("/login", response_model=DataResponse[LoginResponse])
async def login(
    request: Request,
    data: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_async_db),
) -> DataResponse[LoginResponse]:
    """
    Authenticate user.
    """
    try:
        result = await db.execute(
            text("SELECT * FROM authenticate_user_native(:email, :password)"),
            {"email": data.email, "password": data.password},
        )
        row = result.fetchone()

        if not row:
            raise AuthenticationException(message="Invalid email or password")

        user_id, email, tier, _ = row

        await set_user_context(db, str(user_id))
        await db.commit()

        tokens = auth_service.create_token_pair(str(user_id), email, str(tier))
        return DataResponse(
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


@router.post("/refresh", response_model=DataResponse[TokenResponse])
async def refresh_token(
    data: RefreshTokenRequest,
    db: AsyncSession = Depends(get_async_db),
) -> DataResponse[TokenResponse]:
    """
    Refresh access token using a valid refresh token.
    """
    try:
        token_data = auth_service.decode_token(data.refresh_token)
        if token_data.token_type != "refresh":
            raise AuthenticationException(message="Invalid token type")

        if await auth_service.token_blacklist.contains(token_data.jti):
            logger.warning(
                "refresh_token_reuse_detected", jti=token_data.jti, user_id=token_data.user_id
            )
            raise AuthenticationException(message="Token has been revoked")

        await auth_service.token_blacklist.add(token_data.jti, token_data.exp)

        new_tokens = auth_service.create_token_pair(
            token_data.user_id, token_data.email, token_data.tier
        )

        return DataResponse(
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
    return DataResponse(
        data=UserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            tier=str(user.tier),
            is_active=user.is_active,
            is_verified=user.is_verified,
            is_mfa_enabled=user.is_mfa_enabled,
            created_at=user.created_at,
            last_login=user.last_login,
        )
    )


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
        await auth_service.revoke_token(token)
    return SuccessResponse(message="Successfully logged out")


@router.post("/mfa/setup", response_model=DataResponse[MFASetupResponse])
async def mfa_setup(
    response: Response,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
) -> DataResponse[MFASetupResponse]:
    """
    Initialize MFA setup for the user.
    """
    if not user.mfa_secret:
        plain_secret = auth_service.generate_mfa_secret()
        user.mfa_secret = auth_service.encrypt_mfa_secret(plain_secret)
        await db.commit()
    else:
        plain_secret = auth_service.decrypt_mfa_secret(user.mfa_secret)

    uri = auth_service.get_totp_uri(user.email, plain_secret)

    return DataResponse(
        data=MFASetupResponse(
            secret=plain_secret,
            provisioning_uri=uri,
            qr_code_uri=None,
            backup_codes=[],
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

    plain_secret = auth_service.decrypt_mfa_secret(user.mfa_secret)

    if not auth_service.verify_mfa_code(plain_secret, data.code):
        raise AuthenticationException(message="Invalid MFA code")

    user.mfa_enabled = True
    await db.commit()

    return SuccessResponse(message="MFA enabled successfully")


@router.post("/mfa/disable")
async def mfa_disable(
    data: MFAVerifyRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
) -> SuccessResponse:
    """
    Disable MFA for the user.
    """
    if not user.mfa_enabled:
        raise HTTPException(status_code=400, detail="MFA not enabled")

    plain_secret = auth_service.decrypt_mfa_secret(user.mfa_secret)

    if not auth_service.verify_mfa_code(plain_secret, data.code):
        raise AuthenticationException(message="Invalid MFA code")

    user.mfa_enabled = False
    user.mfa_secret = None
    await db.commit()

    return SuccessResponse(message="MFA disabled successfully")


@router.get("/webauthn/register/options")
async def get_webauthn_registration_options(
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
) -> DataResponse[dict]:
    """Generate options for WebAuthn credential registration."""
    options = auth_service.get_webauthn_registration_options(str(user.id), user.email, [])
    return DataResponse(data=options)


@router.post("/webauthn/register/verify")
async def verify_webauthn_registration(
    data: WebAuthnRegistrationVerificationRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
) -> SuccessResponse:
    """Verify and store a new WebAuthn credential."""
    try:
        auth_service.verify_webauthn_registration(
            data.registration_response, data.challenge
        )
        await db.commit()
        return SuccessResponse(message="Passkey registered successfully")
    except Exception as e:
        logger.error(f"webauthn_registration_failed: {e}")
        raise HTTPException(status_code=400, detail="WebAuthn verification failed")


@router.post("/webauthn/login/options")
async def get_webauthn_login_options(
    email: str,
    db: AsyncSession = Depends(get_async_db),
) -> DataResponse[dict]:
    """Generate options for WebAuthn authentication (login)."""
    options = auth_service.get_webauthn_authentication_options([])
    return DataResponse(data=options)


@router.post("/webauthn/login/verify", response_model=DataResponse[LoginResponse])
async def verify_webauthn_login(
    data: WebAuthnAuthenticationVerificationRequest,
    db: AsyncSession = Depends(get_async_db),
) -> DataResponse[LoginResponse]:
    """Verify WebAuthn login and issue tokens."""
    result = await db.execute(select(User).where(User.email == data.email))
    user = result.scalar_one_or_none()
    if not user:
        raise AuthenticationException(message="User not found")

    try:
        from src.database.models import UserCredential
        result_cred = await db.execute(
            select(UserCredential).where(
                UserCredential.user_id == user.id,
                UserCredential.credential_id == data.authentication_response['id']
            )
        )
        cred = result_cred.scalar_one_or_none()
        if not cred:
            raise AuthenticationException(message="Passkey not found for this user")

        import base64
        public_key_bytes = base64.b64decode(cred.public_key)

        auth_service.verify_webauthn_authentication(
            authentication_response=data.authentication_response,
            expected_challenge=data.challenge,
            credential_public_key=public_key_bytes,
            credential_current_sign_count=cred.sign_count,
        )

        cred.sign_count += 1
        cred.last_used_at = datetime.now(UTC)
        await db.commit()

        tokens = auth_service.create_token_pair(str(user.id), user.email, str(user.tier))
        return DataResponse(
            data=LoginResponse(
                access_token=tokens.access_token,
                refresh_token=tokens.refresh_token,
                token_type=tokens.token_type,
                expires_in=tokens.expires_in,
                user_id=str(user.id),
                email=user.email,
                tier=str(user.tier),
            ),
            message="Passkey login successful",
        )
    except Exception as e:
        logger.error(f"webauthn_login_failed: {e}")
        raise AuthenticationException(message="Passkey verification failed")


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
    if not auth_service.verify_password(data.current_password, user.hashed_password):
        raise AuthenticationException(message="Invalid current password")

    user.hashed_password = auth_service.hash_password(data.new_password)
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
        token = auth_service.generate_reset_token()
        user.reset_token = token
        user.reset_token_expires_at = datetime.now(UTC) + timedelta(hours=1)
        await db.commit()

        background_tasks.add_task(_send_password_reset_email, user.email, token)

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

    user.hashed_password = auth_service.hash_password(data.new_password)
    user.reset_token = None
    user.reset_token_expires_at = None
    await db.commit()

    return SuccessResponse(message="Password has been reset successfully")


async def _send_verification_email(email: str, token: str) -> None:
    """
    Sends a verification email to the user.
    """
    verification_link = f"{settings.BETTER_AUTH_URL}/verify-email?token={token}"
    logger.info("email_verification_link_generated", email=email)

    from src.workers.tasks.email_tasks import send_transactional_email

    send_transactional_email.delay(
        to_email=email,
        subject="Verify Your BSOpt Account",
        template_name="verification_email.html",
        context={"verification_link": verification_link, "email": email},
    )


async def _send_password_reset_email(email: str, token: str) -> None:
    """
    Sends a password reset email to the user.
    """
    reset_link = f"{settings.BETTER_AUTH_URL}/reset-password?token={token}"
    logger.info("password_reset_link_generated", email=email)

    from src.workers.tasks.email_tasks import send_transactional_email

    send_transactional_email.delay(
        to_email=email,
        subject="Password Reset Request",
        template_name="password_reset.html",
        context={"reset_link": reset_link, "email": email},
    )
