"""
Authentication Routes (Singularity Refactored)
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, Request, status, HTTPException, Form
from sqlalchemy.orm import Session
from src.database import get_db
from src.database.models import User, OAuth2Client
from src.auth.service import AuthService
from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

async def get_current_user(request: Request):
    if not hasattr(request.state, "user"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return request.state.user

async def get_current_active_user(user = Depends(get_current_user)):
    return user

def _verify_mfa_code(user: User, code: str, db: Session):
    return True # Simulation

async def _send_verification_email(email: str, token: str):
    pass # Simulation

async def _send_password_reset_email(email: str, token: str):
    pass # Simulation

@router.post("/token")
async def login_for_access_token(
    client_id: str = Form(...),
    client_secret: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Standard OAuth 2.0 Token Endpoint.
    Issues a JWT for valid client credentials.
    """
    service = AuthService(db)
    client = service.verify_client(client_id, client_secret)
    
    if not client:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client credentials"
        )
    
    token = service.create_token(client_id, client.scopes or [])
    
    return {
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "scope": " ".join(client.scopes or [])
    }

@router.get("/me")
async def read_users_me(request: Request):
    """
    Returns the current authenticated user's payload.
    """
    if not hasattr(request.state, "user"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return request.state.user

@router.get("/.well-known/openid-configuration")
async def openid_configuration(request: Request):
    base_url = str(request.base_url).rstrip("/")
    return {
        "issuer": f"{base_url}/api/v1/auth",
        "token_endpoint": f"{base_url}/api/v1/auth/token",
        "jwks_uri": f"{base_url}/api/v1/auth/jwks"
    }

@router.get("/jwks")
async def jwks():
    from authlib.jose import JsonWebKey
    key = JsonWebKey.import_key(
        settings.rsa_public_key, 
        {"kty": "RSA", "kid": "internal-key-01", "use": "sig"}
    )
    return {"keys": [key.as_dict()]}
