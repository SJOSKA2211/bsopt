"""
Unified Authentication Service (EquaFlow Phase 2)

Consolidates logic from password.py, oauth2.py, and mfa.py into a high-performance,
zero-trust compliant service.

Features:
- Argon2id password hashing
- TOTP-based MFA
- Asymmetric JWT (ES256/RS256)
- Redis-backed session & revocation
- mTLS header verification support
"""

import hashlib
import logging
import secrets
from datetime import UTC, datetime, timedelta

import jwt
import msgspec
import pyotp
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from cryptography.fernet import Fernet
from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jwt.exceptions import ExpiredSignatureError, PyJWTError
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from src.database import get_async_db
from src.database.models import APIKey, OAuth2Client, User
from src.shared.config import settings
from src.shared.utils.cache import get_redis_client

logger = logging.getLogger(__name__)

# Security schemes for FastAPI docs
security_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class TokenData(BaseModel):
    """Standardized Token Data (Pydantic V2)."""

    user_id: str
    email: str
    tier: str
    token_type: str
    exp: datetime
    iat: datetime
    jti: str | None = None
    scopes: list[str] = []

    model_config = ConfigDict(frozen=True)


class TokenPair(BaseModel):
    """Standardized Token Pair (Pydantic V2)."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    requires_mfa: bool = False


class AuthService:
    """
    Unified Authentication Service.
    """

    def __init__(self):
        self.ph = PasswordHasher(
            time_cost=settings.ARGON2_TIME_COST,
            memory_cost=settings.ARGON2_MEMORY_COST,
            parallelism=settings.ARGON2_PARALLELISM,
        )
        self._fernet = None
        # Pre-computed dummy hash to prevent DoS via repeated Argon2 hashing on invalid usernames
        self.DUMMY_HASH = self.ph.hash("static-dummy-password-for-timing-protection")

    @property
    def fernet(self) -> Fernet:
        """Lazy initialization of Fernet for MFA secret encryption."""
        if self._fernet is None:
            key = settings.MFA_ENCRYPTION_KEY
            if not key:
                raise ValueError("MFA_ENCRYPTION_KEY is missing")
            self._fernet = Fernet(key.encode())
        return self._fernet

    # --- Password Logic (Argon2id) ---

    def hash_password(self, password: str) -> str:
        """Hash a password using Argon2id."""
        return self.ph.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against an Argon2id hash."""
        try:
            return self.ph.verify(hashed_password, plain_password)
        except VerifyMismatchError:
            return False
        except Exception as e:
            logger.error(f"password_verification_error: {e}")
            return False

    def needs_rehash(self, hashed_password: str) -> bool:
        """Check if a hash needs to be updated to current Argon2id parameters."""
        if not hashed_password.startswith("$argon2"):
            return True
        try:
            return self.ph.check_needs_rehash(hashed_password)
        except Exception:
            return True

    # --- MFA Logic (TOTP) ---

    def generate_mfa_secret(self) -> str:
        """Generate a new TOTP secret."""
        return pyotp.random_base32()

    def encrypt_mfa_secret(self, secret: str) -> str:
        """Encrypt MFA secret for database storage."""
        return self.fernet.encrypt(secret.encode()).decode()

    def decrypt_mfa_secret(self, encrypted_secret: str) -> str:
        """Decrypt MFA secret for verification."""
        return self.fernet.decrypt(encrypted_secret.encode()).decode()

    def get_totp_uri(self, email: str, secret: str) -> str:
        """Generate a provisioning URI for QR codes."""
        return pyotp.totp.TOTP(secret).provisioning_uri(name=email, issuer_name="EquaFlow")

    def verify_mfa_code(self, secret: str, code: str) -> bool:
        """Verify a TOTP code with clock skew support."""
        if not secret or not code:
            return False
        totp = pyotp.TOTP(secret)
        return totp.verify(code, valid_window=1)

    # --- Token Generation Helpers ---

    def generate_reset_token(self) -> str:
        """Generate a secure, high-entropy reset token."""
        return secrets.token_urlsafe(32)

    def generate_verification_token(self) -> str:
        """Generate a secure, high-entropy verification token."""
        return secrets.token_urlsafe(32)

    # --- Token Logic (JWT ES256/RS256) ---

    def _get_key_for_algorithm(self, algorithm: str, is_private: bool = True) -> str:
        """Selects the correct key (RSA or ECC) based on the algorithm."""
        if algorithm.startswith("RS"):
            return settings.rsa_private_key if is_private else settings.rsa_public_key
        elif algorithm.startswith("ES"):
            return settings.es256_private_key if is_private else settings.es256_public_key

        if settings.is_production:
            raise ValueError(f"Symmetric algorithm {algorithm} forbidden in production.")
        return settings.JWT_SECRET

    def create_token_pair(
        self, user_id: str, email: str, tier: str, scopes: list[str] = []
    ) -> TokenPair:
        """Create a pair of access and refresh tokens."""
        access_token = self._create_token(
            {"sub": user_id, "email": email, "tier": tier, "type": "access", "scopes": scopes},
            timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        )
        refresh_token = self._create_token(
            {"sub": user_id, "email": email, "tier": tier, "type": "refresh", "scopes": scopes},
            timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        )
        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    def _create_token(self, data: dict, expires_delta: timedelta) -> str:
        """Internal helper to create a JWT token with asymmetric support and strict msgspec validation."""
        now = datetime.now(UTC)
        expire = now + expires_delta

        # Institutional Payload with strict entropy
        to_encode = {
            **data,
            "exp": expire,
            "iat": now,
            "jti": secrets.token_hex(24),  # Expanded entropy for institutional security
            "iss": "equaflow-manifold-v2",
        }

        # Use msgspec for fast serialization check (ensures no non-JSON serializable objects)
        msgspec.json.encode(to_encode)

        algorithm = settings.JWT_ALGORITHM
        key = self._get_key_for_algorithm(algorithm, is_private=True)
        return jwt.encode(to_encode, key, algorithm=algorithm)

    def decode_token(self, token: str) -> TokenData:
        """Decode and validate a JWT token."""
        try:
            unverified_header = jwt.get_unverified_header(token)
            algorithm = unverified_header.get("alg", settings.JWT_ALGORITHM)
            key = self._get_key_for_algorithm(algorithm, is_private=False)
            payload = jwt.decode(token, key, algorithms=[algorithm])

            return TokenData(
                user_id=payload.get("sub"),
                email=payload.get("email", ""),
                tier=payload.get("tier", "free"),
                token_type=payload.get("type", "access"),
                exp=datetime.fromtimestamp(payload.get("exp", 0), tz=UTC),
                iat=datetime.fromtimestamp(payload.get("iat", 0), tz=UTC),
                jti=payload.get("jti"),
                scopes=payload.get("scopes", []),
            )
        except ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    # --- Session & Revocation (Redis) ---

    async def is_token_revoked(self, jti: str) -> bool:
        """Check if a token JTI is in the blacklist."""
        redis = await get_redis_client()
        return bool(await redis.exists(f"blacklist:{jti}"))

    async def revoke_token(self, token: str) -> None:
        """Revoke a token by adding its JTI to the blacklist."""
        try:
            token_data = self.decode_token(token)
            if token_data.jti:
                redis = await get_redis_client()
                ttl = int((token_data.exp - datetime.now(UTC)).total_seconds())
                if ttl > 0:
                    await redis.setex(f"blacklist:{token_data.jti}", ttl, "1")
        except Exception as e:
            logger.warning(f"token_revocation_failed: {e}")

    # --- Core Flow ---

    async def authenticate_user(self, db: AsyncSession, email: str, password: str) -> User:
        """
        Authenticate a user by email and password.
        Implements timing attack protection and automatic re-hashing.
        """
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if not user:
            # Timing attack protection: burn CPU time even if user not found
            await run_in_threadpool(self.verify_password, password, self.DUMMY_HASH)
            raise HTTPException(status_code=401, detail="Invalid credentials")

        if not user.hashed_password:
            # Handle users without passwords (e.g. OAuth only)
            await run_in_threadpool(self.verify_password, password, self.DUMMY_HASH)
            raise HTTPException(status_code=401, detail="Invalid credentials")

        password_matches = await run_in_threadpool(
            self.verify_password, password, user.hashed_password
        )
        if not password_matches:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        if self.needs_rehash(user.hashed_password):
            logger.info("password_rehash_triggered", user_id=str(user.id))
            user.hashed_password = self.hash_password(password)
            # Persisted by the commit in the calling route

        return user

    async def verify_mfa(self, user: User, code: str | None) -> bool:
        """Verify MFA code if enabled for the user."""
        if not user.mfa_enabled:
            return True

        if not code:
            return False

        try:
            secret = self.decrypt_mfa_secret(user.mfa_secret)
            return self.verify_mfa_code(secret, code)
        except Exception as e:
            logger.error(f"mfa_verification_error: {e}")
            return False

    async def validate_token(self, token: str) -> TokenData:
        """
        High-performance token validation.
        Checks Redis session cache first, then JWT signature and revocation.
        """
        # 1. Redis Session Cache (Fast Path)
        redis = await get_redis_client()
        try:
            cached_data = await redis.get(f"session_v2:{token}")
            if cached_data:
                data = msgspec.json.decode(cached_data)
                return TokenData(**data)
        except Exception as e:
            logger.debug("session_cache_lookup_failed", error=str(e))

        # 2. Asymmetric JWT Validation
        token_data = self.decode_token(token)

        # 3. Revocation Check
        if token_data.jti and await self.is_token_revoked(token_data.jti):
            raise HTTPException(status_code=401, detail="Token revoked")

        # 4. Cache Session for future requests
        try:
            ttl = int((token_data.exp - datetime.now(UTC)).total_seconds())
            if ttl > 0:
                await redis.setex(
                    f"session_v2:{token}",
                    ttl,
                    msgspec.json.encode(token_data.model_dump(mode="json")),
                )
        except Exception as e:
            logger.warning("session_cache_write_failed", error=str(e))

        return token_data

    # --- OAuth2 Client Logic ---

    async def authenticate_client(
        self, db: AsyncSession, client_id: str, client_secret: str
    ) -> OAuth2Client:
        """Authenticate a confidential OAuth2 client."""
        result = await db.execute(select(OAuth2Client).where(OAuth2Client.client_id == client_id))
        client = result.scalar_one_or_none()

        if not client:
            # Dummy comparison to prevent timing attacks
            secrets.compare_digest("dummy", client_secret)
            raise HTTPException(status_code=401, detail="Invalid client credentials")

        if not secrets.compare_digest(client.client_secret, client_secret):
            raise HTTPException(status_code=401, detail="Invalid client credentials")

        return client

    def create_client_credentials_token(self, client: OAuth2Client, scopes: list[str]) -> TokenPair:
        """Create a token for Client Credentials flow."""
        allowed_scopes = set(client.scopes or [])
        requested_scopes = set(scopes)
        if not requested_scopes.issubset(allowed_scopes):
            raise HTTPException(status_code=400, detail="Invalid scope requested")

        access_token = self._create_token(
            {"sub": client.client_id, "type": "client_credentials", "scopes": scopes},
            timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        )
        return TokenPair(
            access_token=access_token,
            refresh_token="",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    # --- mTLS Support ---

    def verify_mtls(self, request: Request) -> bool:
        """
        Verify mTLS headers from trusted proxies.
        Bit-perfect compatibility with X-SSL-Client-Verify expectations.
        """
        client_verify = request.headers.get("X-SSL-Client-Verify")
        if client_verify != "SUCCESS":
            logger.warning("mtls_verification_failed", status=client_verify)
            return False

        return True


# Global instance
auth_service = AuthService()


def get_auth_service() -> AuthService:
    return auth_service


# --- FastAPI Dependencies ---


async def get_token_from_header(
    credentials: HTTPAuthorizationCredentials | None = Depends(security_scheme),
) -> str | None:
    if credentials:
        return credentials.credentials
    return None


async def get_current_user(
    request: Request,
    token: str | None = Depends(get_token_from_header),
    db: AsyncSession = Depends(get_async_db),
    service: AuthService = Depends(get_auth_service),
) -> User:
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    token_data = await service.validate_token(token)
    user_id = token_data.user_id

    # Try cache first
    from src.shared.utils.cache import db_cache

    try:
        cached_user = await db_cache.get_user(user_id)
        if cached_user:
            user = User(**cached_user)
            request.state.user = user
            return user
    except Exception:
        pass

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    request.state.user = user
    return user


async def get_current_active_user(
    user: User = Depends(get_current_user),
) -> User:
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")
    return user


class RoleChecker:
    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = allowed_roles

    async def __call__(self, user: User = Depends(get_current_active_user)):
        user_roles = [user.tier]
        if user.tier == "enterprise":
            user_roles.append("admin")

        if not set(self.allowed_roles).intersection(user_roles):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user


async def get_api_key(
    request: Request,
    api_key: str | None = Depends(api_key_header),
    db: AsyncSession = Depends(get_async_db),
) -> User | None:
    if not api_key:
        return None

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    result = await db.execute(select(APIKey).where(APIKey.key_hash == key_hash, APIKey.is_active))
    key_record = result.scalar_one_or_none()

    if not key_record:
        return None

    key_record.last_used_at = datetime.now(UTC)
    await db.commit()

    return key_record.user


async def get_current_user_flexible(
    request: Request,
    token: str | None = Depends(get_token_from_header),
    api_key_user: User | None = Depends(get_api_key),
    db: AsyncSession = Depends(get_async_db),
    service: AuthService = Depends(get_auth_service),
) -> User | None:
    if api_key_user:
        return api_key_user

    if token:
        try:
            return await get_current_user(request, token, db, service)
        except HTTPException:
            pass

    return None
