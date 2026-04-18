"""Authentication and authorization utilities."""

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import redis.asyncio
from passlib.context import CryptContext

from src.shared.config import settings

logger = logging.getLogger(__name__)

# --- Redis Configuration for Token Revocation ---
REDIS_URL = settings.REDIS_URL
redis_pool = redis.asyncio.ConnectionPool.from_url(REDIS_URL)
redis_client = redis.asyncio.Redis(connection_pool=redis_pool)

# --- Password Hashing ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- JWT Configuration ---
JWT_SECRET_KEY = settings.JWT_SECRET
JWT_ALGORITHM = settings.JWT_ALGORITHM

async def add_revoked_token(token: str, expires_at: datetime) -> None:
    """Add a token to the Redis revocation list with an expiration time."""
    if redis_client:
        # Expiration is set as a Unix timestamp (seconds since epoch).
        await redis_client.set(
            f"revoked_token:{token}",
            "true",
            exat=int(expires_at.timestamp()),
        )

async def is_token_revoked(token: str) -> bool:
    """Check if a token is in the Redis revocation list."""
    if redis_client:
        is_revoked = await redis_client.get(f"revoked_token:{token}")
        return is_revoked is not None
    return False

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password."""
    return bool(pwd_context.verify(plain_password, hashed_password))

def get_password_hash(password: str) -> str:
    """Generate a bcrypt hash of a password."""
    return str(pwd_context.hash(password))

def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Create a new JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=15)
    to_encode.update({"exp": int(expire.timestamp()), "iat": int(datetime.now(UTC).timestamp())})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def create_refresh_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Create a new JWT refresh token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(days=30)
    to_encode.update({"exp": int(expire.timestamp()), "iat": int(datetime.now(UTC).timestamp())})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

async def verify_token(token: str) -> dict[str, Any] | None:
    """Verify a JWT token and return its payload if valid."""
    try:
        # First, check if the token is in our revocation list
        if await is_token_revoked(token):
            return None

        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

        token_exp_timestamp = payload.get("exp")
        if token_exp_timestamp is None:
            return None

        token_exp_dt = datetime.fromtimestamp(token_exp_timestamp, tz=UTC)
        if datetime.now(UTC) > token_exp_dt:
            return None # Token expired

        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None
    except Exception:
        logger.exception("Unexpected error during token verification")
        return None

async def revoke_token(token: str) -> None:
    """Add a token to the revocation list in Redis."""
    payload = await verify_token(token)
    if payload and "exp" in payload:
        exp = datetime.fromtimestamp(payload["exp"], tz=UTC)
        await add_revoked_token(token, exp)
    elif redis_client:
        # If we can't find the expiry, set a default TTL of 24 hours
        await redis_client.set(f"revoked_token:{token}", "true", ex=86400)

async def create_mfa_challenge(user_id: str) -> dict[str, Any]:
    """Generate a placeholder MFA challenge."""
    session_id = f"mfa_session_{uuid.uuid4()}"
    return {
        "user_id": user_id,
        "session_id": session_id,
        "method": "totp",
        "challenge": "Verify with your authenticator app",
    }

def verify_mfa_response(session_id: str, response_code: str) -> bool:
    """Verify a placeholder MFA response."""
    logger.info("Mock verifying MFA for session %s with code %s", session_id, response_code)
    return True
