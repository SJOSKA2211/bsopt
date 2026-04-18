import os
import uuid  # For generating IDs
from datetime import UTC, datetime, timedelta
from typing import Any  # Import Set for revoked tokens

import jwt

# --- Redis Client Setup ---
import redis  # Import the redis library
from passlib.context import CryptContext

# Use REDIS_URL from environment variables, defaulting to localhost if not set
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# Ensure REDIS_PASSWORD is provided in .env and used in REDIS_URL
# If REDIS_URL does not contain password, it might lead to connection issues.
# The bootstrap script sets REDIS_URL=redis://:$REDIS_PASS@redis:6379/0, which is good.

try:
    # Use connection pool for efficiency
    redis_pool = redis.asyncio.ConnectionPool.from_url(REDIS_URL, decode_responses=True)
    redis_client = redis.asyncio.Redis(connection_pool=redis_pool)
    # Ping Redis to ensure connection is established early
    async def ping_redis():
        await redis_client.ping()
        # logger.info("Successfully connected to Redis.") # This would require logger to be available here
    # Running ping_redis() directly here will block module import if not handled asynchronously.
    # A better approach is to ensure connection happens on first use or during app startup.
    # For now, we assume connection will be established when first used.
except redis.exceptions.ConnectionError:
    # Handle connection errors during import if necessary, or rely on first use to fail.
    # logger.error(f"Could not connect to Redis at {REDIS_URL}: {e}") # Requires logger
    redis_client = None # Set to None if connection fails at import time
    # In a real app, you'd want more robust error handling or retry logic.

# --- Configuration ---
JWT_SECRET_KEY = os.getenv("JWT_SECRET", "my-super-secret-key-for-development-only")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "RS256") # Production standard, use HS256 for simplicity if keys aren't setup
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))
MFA_ENCRYPTION_KEY = os.getenv("MFA_ENCRYPTION_KEY", "my-mfa-secret-key-for-development-only")

# --- Token Revocation Store ---
# Replaced in-memory set with Redis client for persistence.
# Tokens are stored with an expiration time matching their JWT expiry.

async def add_revoked_token(token: str, expires_at: datetime):
    """Adds a token to the Redis revocation list with an expiration time."""
    if redis_client:
        # Store the token in Redis with an expiration time.
        # Expiration is set as a Unix timestamp (seconds since epoch).
        await redis_client.set(f"revoked_token:{token}", "true", exat=expires_at.timestamp())
        # logger.info(f"Token added to revocation list: {token[:10]}...") # Requires logger
    else:
        # logger.warning("Redis client not available, cannot add token to revocation list.") # Requires logger
        pass # Cannot revoke if Redis is unavailable

async def is_token_revoked(token: str) -> bool:
    """Checks if a token is in the Redis revocation list."""
    if redis_client:
        is_revoked = await redis_client.get(f"revoked_token:{token}")
        # logger.debug(f"Checking revocation for token {token[:10]}... Result: {is_revoked}") # Requires logger
        return is_revoked is not None
    # logger.warning("Redis client not available, cannot check token revocation.") # Requires logger
    return False # Assume not revoked if Redis is down, or handle more gracefully

# --- Password Hashing Context ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "iat": datetime.now(UTC)})

    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "iat": datetime.now(UTC)})

    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

async def verify_token(token: str) -> dict[str, Any] | None:
    try:
        # First, check if the token is in our revocation list
        if await is_token_revoked(token):
            # logger.warning("Token is revoked.") # Requires logger
            return None

        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

        # Check token expiry
        token_exp_timestamp = payload.get("exp")
        if token_exp_timestamp is None:
            # logger.warning("Token missing expiry time.") # Requires logger
            return None

        # Convert JWT expiry (usually float seconds) to datetime object for comparison
        token_exp_dt = datetime.fromtimestamp(token_exp_timestamp, tz=UTC)

        if datetime.now(UTC) > token_exp_dt:
            # logger.warning("Token has expired.") # Requires logger
            return None # Token expired

        return payload
    except jwt.ExpiredSignatureError:
        # logger.warning("Token signature has expired.") # Requires logger
        return None
    except jwt.InvalidTokenError:
        # logger.warning(f"Invalid token: {e}") # Requires logger
        return None
    except Exception:
        # Catch any other unexpected errors during token verification
        # logger.error(f"An unexpected error occurred during token verification: {e}") # Requires logger
        return None

async def revoke_token(token: str):
    """Adds a token to the revocation list in Redis."""
    # Retrieve token expiry from payload to set Redis TTL correctly
    payload = await verify_token(token) # Re-verify to get payload and expiry if token is still valid
    if payload and "exp" in payload:
        expires_at = datetime.fromtimestamp(payload["exp"], tz=UTC)
        await add_revoked_token(token, expires_at)
    # If token is already expired or invalid, we might not get expiry.
    # For simplicity, we can still mark it as revoked, but it won't have a specific expiry in Redis.
    # A more robust solution might be to handle this differently or use a fixed TTL.
    # logger.warning(f"Could not determine expiry for token {token[:10]} to set revocation TTL. Revoking without expiry.") # Requires logger
    elif redis_client:
        await redis_client.set(f"revoked_token:{token}", "true") # No expiry, might accumulate

async def create_mfa_challenge(user_id: str) -> dict[str, Any]:
    """Placeholder for MFA challenge generation."""
    session_id = f"mfa_session_{uuid.uuid4()}" # Use UUID for session IDs
    # In a real system, this would involve generating a TOTP secret or similar
    # and encrypting challenge data.
    return {
        "method": "TOTP",
        "challenge": "some_encrypted_challenge_data", # Encrypt actual challenge data with MFA_ENCRYPTION_KEY
        "session_id": session_id,
    }

def verify_mfa_response(session_id: str, response_code: str) -> bool:
    """Placeholder for MFA response verification."""
    # logger.info(f"Verifying MFA for session {session_id} with code {response_code}") # Requires logger
    # Actual verification logic would go here
    return True
