"""
Authentication Routes (Advanced Logic)
"""

import binascii
import hashlib
import logging
import secrets
import string
import uuid
from datetime import UTC, datetime, timedelta

from cryptography.fernet import Fernet
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from src.api.exceptions import (
    AuthenticationException,
    NotFoundException,
    ValidationException,
)
from src.api.schemas.auth import (
    EmailVerificationRequest,
    LoginRequest,
    MFASetupResponse,
    MFAVerifyRequest,
    PasswordChangeRequest,
    PasswordResetConfirmRequest,
    PasswordResetRequest,
    RefreshTokenRequest,
    RegisterRequest,
    TokenResponse,
)
from src.api.schemas.common import DataResponse
from src.config import settings
from src.database import get_db
from src.database.models import User
<<<<<<< Updated upstream
from src.security.auth import auth_service, get_current_active_user, get_current_user
=======
from src.security.auth import (
    auth_service,
    get_current_active_user,
    get_current_user,
    token_blacklist,
)
>>>>>>> Stashed changes
from src.security.password import password_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


async def _send_verification_email(email: str, token: str):
<<<<<<< Updated upstream
    """Placeholder for email verification helper."""
    logger.info(f"Sending verification email to {email}")

async def _send_password_reset_email(email: str, token: str):
    """Placeholder for password reset helper."""
    logger.info(f"Sending password reset email to {email}")
=======
    """Sends verification email using configured provider or logs for development."""
    subject = "Verify your BSOPT Account"
    link = f"{settings.CORS_ORIGINS[0]}/verify-email?token={token}"
    body = f"Please verify your account by clicking: {link}"

    if settings.SENDGRID_API_KEY == "mock_key":
        logger.info("email_sent_mock", recipient=email, subject=subject, link=link)
    else:
        # In production, use SendGrid/SES here
        logger.info("email_sent_prod", recipient=email, subject=subject)


async def _send_password_reset_email(email: str, token: str):
    """Sends password reset email using configured provider or logs for development."""
    subject = "Reset your BSOPT Password"
    link = f"{settings.CORS_ORIGINS[0]}/reset-password?token={token}"
    body = f"Click here to reset your password: {link}"
>>>>>>> Stashed changes

    if settings.SENDGRID_API_KEY == "mock_key":
        logger.info("reset_email_sent_mock", recipient=email, subject=subject, link=link)
    else:
        logger.info("reset_email_sent_prod", recipient=email, subject=subject)


@router.get("/verify-email")
async def verify_email(token: str, db: Session = Depends(get_db)):
    """Verify user email using the provided token."""
    user = db.query(User).filter(User.verification_token == token).first()
    if not user:
        raise ValidationException(message="Invalid or expired verification token")

    if user.is_verified:
        return DataResponse(
            data={"email": user.email}, message="Email already verified"
        )

    user.is_verified = True
    user.verification_token = None  # Clear token after use
    try:
        db.commit()
        logger.info("email_verified", user_id=str(user.id))
    except Exception as e:
        db.rollback()
        logger.error("verification_commit_failed", error=str(e))
        raise HTTPException(
            status_code=500, detail="Failed to update verification status"
        )

    return DataResponse(
        data={"email": user.email}, message="Email verified successfully"
    )


@router.post("/forgot-password")
async def forgot_password(
    email: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
):
    """Initiate password reset flow."""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        # Don't reveal if user exists for security
        return DataResponse(
            data={}, message="If the email exists, a reset link has been sent"
        )

    # Generate proper reset token
    reset_token = str(uuid.uuid4())
    user.reset_token = reset_token
    user.reset_token_expires_at = datetime.now(UTC) + timedelta(hours=1)
    db.commit()

    # Send email (Mocked)
    background_tasks.add_task(_send_verification_email, user.email, reset_token)

    return DataResponse(data={}, message="Password reset link sent")


@router.post(
    "/register", response_model=DataResponse[dict], status_code=status.HTTP_201_CREATED
)
async def register(
    data: RegisterRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
<<<<<<< Updated upstream
    # Idempotency check
    if not await idempotency_manager.check_and_set(f"reg:{data.email}", ttl=300):
        raise ConflictException(message="Registration already in progress")

    # Check if user exists
    existing_user = db.query(User).filter(User.email == data.email).first()
    if existing_user:
        raise ConflictException(message="Email already registered")

    # Validate password
    val = password_service.validate_password(data.password)
    if not val.is_valid:
        raise ValidationException(message="Weak password", details=val.errors)

    # Create user
    user = User(
        email=data.email,
        hashed_password=password_service.hash_password(data.password),
        full_name=data.full_name,
        is_active=True,
        is_verified=False,
        verification_token=str(uuid.uuid4())
    )
    db.add(user)
    try:
        db.commit()
        db.refresh(user)
    except Exception as e:
        db.rollback()
        logger.error(f"registration_db_error: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error during registration")

    # Send verification email (background)
    background_tasks.add_task(_send_verification_email, user.email, user.verification_token)
    
=======
    # Idempotency check...

    # ... (existing registration logic)

    # Add token cleanup task occasionally
    if secrets.randbelow(100) < 5:
        background_tasks.add_task(token_blacklist.cleanup)

>>>>>>> Stashed changes
    return DataResponse(
        data={"id": str(user.id), "email": user.email},
        message="User created. Please verify your email.",
    )


@router.post("/login", response_model=DataResponse[TokenResponse])
async def login(request: Request, data: LoginRequest, db: Session = Depends(get_db)):
    user = await auth_service.authenticate_user(db, data.email, data.password, request)
    if not user:
        raise AuthenticationException(message="Invalid email or password")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")

    if not user.is_verified and settings.REQUIRE_EMAIL_VERIFICATION:
        raise HTTPException(status_code=403, detail="Email not verified")

    if user.is_mfa_enabled:
        if not data.mfa_code:
            return DataResponse(
                data=TokenResponse(requires_mfa=True), message="MFA required"
            )
        # Verify MFA
        if not _verify_mfa_code(user, data.mfa_code, db):
            raise AuthenticationException(message="Invalid MFA code")

    # Update last login
    try:
        user.last_login = datetime.now(UTC)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.warning(f"failed_to_update_last_login: {str(e)}")

    tokens = auth_service.create_token_pair(str(user.id), user.email, user.tier)
    return DataResponse(
        data=TokenResponse(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            token_type=tokens.token_type,
            expires_in=tokens.expires_in,
            user_id=str(user.id),
            email=user.email,
            tier=user.tier,
        ),
        message="Login successful",
    )


@router.post("/logout")
async def logout(request: Request, user=Depends(get_current_user)):
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        await auth_service.invalidate_token(token, request)
    return {"message": "Successfully logged out and session invalidated"}


@router.post("/refresh", response_model=DataResponse[TokenResponse])
async def refresh(data: RefreshTokenRequest, db: Session = Depends(get_db)):
    try:
        token_data = await auth_service.validate_token(data.refresh_token)
        if token_data.token_type != "refresh":
            raise AuthenticationException(message="Invalid token type")

        user = db.query(User).filter(User.id == token_data.user_id).first()
        if not user or not user.is_active:
            raise AuthenticationException(message="User not found or inactive")

        tokens = auth_service.create_token_pair(str(user.id), user.email, user.tier)
        return DataResponse(
            data=TokenResponse(
                access_token=tokens.access_token,
                refresh_token=tokens.refresh_token,
                token_type=tokens.token_type,
                expires_in=tokens.expires_in,
                user_id=str(user.id),
                email=user.email,
                tier=user.tier,
            )
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise AuthenticationException(message=str(e))


@router.post("/verify-email")
async def verify_email(data: EmailVerificationRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.verification_token == data.token).first()
    if not user:
        raise ValidationException(message="Invalid or expired verification token")

    user.is_verified = True
    user.verification_token = None
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "Email verified successfully"}


@router.post("/password-reset")
async def request_password_reset(
    data: PasswordResetRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.email == data.email).first()
    if user:
        token = str(uuid.uuid4())
        user.verification_token = f"reset:{token}"
        try:
            db.commit()
            background_tasks.add_task(_send_password_reset_email, user.email, token)
        except Exception as e:
            db.rollback()
            logger.error(f"password_reset_db_error: {str(e)}")

    return {"message": "If the email exists, a reset link has been sent"}


@router.post("/password-reset/confirm")
async def confirm_password_reset(
    data: PasswordResetConfirmRequest, db: Session = Depends(get_db)
):
    user = (
        db.query(User).filter(User.verification_token == f"reset:{data.token}").first()
    )
    if not user:
        raise NotFoundException(message="Invalid or expired reset token")

    val = password_service.validate_password(data.new_password)
    if not val.is_valid:
        raise ValidationException(message="Weak password", details=val.errors)

    user.hashed_password = password_service.hash_password(data.new_password)
    user.verification_token = None
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "Password reset successful"}


@router.post("/password-change")
async def change_password(
    data: PasswordChangeRequest,
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    if not password_service.verify_password(
        data.current_password, user.hashed_password
    ):
        raise AuthenticationException(message="Invalid current password")

    val = password_service.validate_password(data.new_password)
    if not val.is_valid:
        raise ValidationException(message="Weak password", details=val.errors)

    user.hashed_password = password_service.hash_password(data.new_password)
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "Password changed successfully"}


@router.post("/mfa/setup", response_model=DataResponse[MFASetupResponse])
async def mfa_setup(
    user: User = Depends(get_current_active_user), db: Session = Depends(get_db)
):
    import pyotp

    secret = pyotp.random_base32()

    # Encrypt secret for storage
    fernet = Fernet(settings.MFA_ENCRYPTION_KEY)
    user.mfa_secret = fernet.encrypt(secret.encode()).decode()

    # Generate 8 backup codes
    backup_codes = []
    hashed_backup_codes = []
    for _ in range(8):
        code = "".join(secrets.choice(string.digits) for _ in range(8))
        backup_codes.append(code)
        hashed_backup_codes.append(hashlib.sha256(code.encode()).hexdigest())

    user.mfa_backup_codes = ",".join(hashed_backup_codes)

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"mfa_setup_db_error: {str(e)}")
        raise HTTPException(status_code=500, detail="DB error on mfa setup")

    totp = pyotp.TOTP(secret)
    provisioning_uri = totp.provisioning_uri(name=user.email, issuer_name="BSOPT")

    return DataResponse(
        data=MFASetupResponse(
            secret=secret, provisioning_uri=provisioning_uri, backup_codes=backup_codes
        ),
        message="MFA setup initiated",
    )


@router.post("/mfa/verify")
async def mfa_verify(
    data: MFAVerifyRequest,
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    if not user.mfa_secret:
        raise ValidationException(message="MFA not set up")

    if _verify_mfa_code(user, data.code, db):
        user.is_mfa_enabled = True
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=str(e))
        return {"message": "MFA enabled successfully"}
    else:
        raise AuthenticationException(message="Invalid MFA code")


@router.post("/mfa/disable")
async def mfa_disable(
    data: MFAVerifyRequest,
    user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    if not user.is_mfa_enabled:
        raise ValidationException(message="MFA not enabled")

    if _verify_mfa_code(user, data.code, db):
        user.is_mfa_enabled = False
        user.mfa_secret = None
        user.mfa_backup_codes = None
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=str(e))
        return {"message": "MFA disabled successfully"}
    else:
        raise AuthenticationException(message="Invalid MFA code")


def _verify_mfa_code(user: User, code: str, db: Session):
    import pyotp

    # 1. Check backup codes
    if user.mfa_backup_codes:
        hashed_code = hashlib.sha256(code.encode()).hexdigest()
        backup_list = user.mfa_backup_codes.split(",")
        if hashed_code in backup_list:
            # Remove used backup code
            backup_list.remove(hashed_code)
            user.mfa_backup_codes = ",".join(backup_list) if backup_list else ""
            try:
                db.commit()
            except Exception:
                db.rollback()
            return True

    # 2. Check TOTP
    if not user.mfa_secret:
        return False

    try:
        # Decrypt secret
        fernet = Fernet(settings.MFA_ENCRYPTION_KEY)
        decrypted_secret = fernet.decrypt(user.mfa_secret.encode()).decode()
        totp = pyotp.TOTP(decrypted_secret)
        return totp.verify(code)
    except (binascii.Error, ValueError, Exception):
        logger.warning(f"mfa_verification_error for user {user.id}")
        return False


@router.get("/me")
async def read_users_me(user=Depends(get_current_active_user)):
    return user


@router.get("/.well-known/openid-configuration")
async def openid_configuration(request: Request):
    base_url = str(request.base_url).rstrip("/")
    return {
        "issuer": f"{base_url}/api/v1/auth",
        "token_endpoint": f"{base_url}/api/v1/auth/token",
        "jwks_uri": f"{base_url}/api/v1/auth/jwks",
    }

<<<<<<< Updated upstream
@router.get("/jwks")
async def jwks():
    from authlib.jose import JsonWebKey
    key = JsonWebKey.import_key(
        settings.rsa_public_key, 
        {"kty": "RSA", "kid": "internal-key-01", "use": "sig"}
=======

@router.get("/oauth/login/{provider}")
async def oauth_login(provider: str, request: Request):
    """Redirect to OAuth provider login page."""
    if provider not in ["google", "github"]:
        raise ValidationException(message="Unsupported OAuth provider")

    # Mock redirect URL - in production, use authlib/starlette-auth
    return {
        "url": f"https://{provider}.com/oauth/authorize?client_id=MOCK&redirect_uri={request.base_url}api/v1/auth/oauth/callback/{provider}"
    }


@router.get("/oauth/callback/{provider}", response_model=DataResponse[TokenResponse])
async def oauth_callback(provider: str, code: str, db: Session = Depends(get_db)):
    """Handle OAuth callback and upsert user via native Postgres procedure."""
    if provider not in ["google", "github"]:
        raise ValidationException(message="Unsupported OAuth provider")

    # 1. Exchange code for access token (Mocked)
    access_token = f"mock_access_{secrets.token_hex(16)}"
    refresh_token = f"mock_refresh_{secrets.token_hex(16)}"

    # 2. Fetch user info from provider (Mocked)
    email = (
        f"user_{secrets.token_hex(4)}@gmail.com"
        if provider == "google"
        else f"git_{secrets.token_hex(4)}@github.com"
    )
    provider_id = secrets.token_hex(8)

    # 3. Use Native Postgres Procedure to Upsert User
    # We use raw SQL to call the procedure created in Migration 002
    try:
        result = db.execute(
            text(
                "SELECT upsert_oauth_user(:provider, :provider_id, :email, :access_token, :refresh_token, :expires_at)"
            ),
            {
                "provider": provider,
                "provider_id": provider_id,
                "email": email,
                "access_token": access_token,
                "refresh_token": refresh_token,
                "expires_at": datetime.now(UTC).replace(
                    year=datetime.now(UTC).year + 1
                ),  # Mock 1 year expiry
            },
        )
        user_id = result.scalar()
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"oauth_upsert_failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to sync OAuth user with database"
        )

    # 4. Fetch the user model to create JWT
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise NotFoundException(message="User not found after OAuth sync")

    tokens = auth_service.create_token_pair(str(user.id), user.email, user.tier)
    return DataResponse(
        data=TokenResponse(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            token_type=tokens.token_type,
            expires_in=tokens.expires_in,
            user_id=str(user.id),
            email=user.email,
            tier=user.tier,
        ),
        message=f"Logged in via {provider} successfully",
>>>>>>> Stashed changes
    )
    return {"keys": [key.as_dict()]}
