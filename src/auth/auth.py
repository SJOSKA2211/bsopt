import os
import time
import bcrypt
import jwt
from jwt.algorithms import RSAAlgorithm
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Set # Import Set for revoked tokens
import uuid # For generating IDs

# --- Configuration ---
JWT_SECRET_KEY = os.getenv("JWT_SECRET", "my-super-secret-key-for-development-only")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "RS256") # Production standard, use HS256 for simplicity if keys aren't setup
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))
MFA_ENCRYPTION_KEY = os.getenv("MFA_ENCRYPTION_KEY", "my-mfa-secret-key-for-development-only")

# --- Token Revocation Store ---
# In a real application, this would be a persistent store (e.g., Redis, DB table)
# For simplicity, we use a set in memory. This is NOT production-ready.
REVOKED_TOKENS: Set[str] = set() 

# --- Password Hashing Context ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        if token in REVOKED_TOKENS:
            logger.warning("Token is revoked.")
            return None
            
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        if datetime.now(timezone.utc) > payload.get("exp", datetime.min.replace(tzinfo=timezone.utc)):
            logger.warning("Token has expired.")
            return None # Token expired
        
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token signature has expired.")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        return None

def revoke_token(token: str):
    """Adds a token to the revocation list."""
    REVOKED_TOKENS.add(token)
    logger.info(f"Token revoked: {token[:10]}...")

def create_mfa_challenge(user_id: str) -> Dict[str, Any]:
    """Placeholder for MFA challenge generation."""
    session_id = f"mfa_session_{uuid.uuid4()}" # Use UUID for session IDs
    # In a real system, this would involve generating a TOTP secret or similar
    # and encrypting challenge data.
    return {
        "method": "TOTP",
        "challenge": "some_encrypted_challenge_data", # Encrypt actual challenge data with MFA_ENCRYPTION_KEY
        "session_id": session_id
    }

def verify_mfa_response(session_id: str, response_code: str) -> bool:
    """Placeholder for MFA response verification."""
    logger.info(f"Verifying MFA for session {session_id} with code {response_code}")
    # Actual verification logic would go here
    return True
