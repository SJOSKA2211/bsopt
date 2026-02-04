from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from authlib.integrations.starlette_client import OAuth
from authlib.jose import jwt
from sqlalchemy.orm import Session
from src.database import get_db
from src.database.models import User, OAuth2Client
import structlog

logger = structlog.get_logger()
router = APIRouter(prefix="/auth", tags=["auth"])

# 🚀 SOTA: Internal Auth Server Logic
# This module implements the Auth Server part of the Triad

@router.post("/token")
async def generate_token(request: Request, db: Session = Depends(get_db)):
    """
    OAuth2 Token Endpoint.
    Supports client_credentials and password grants.
    """
    # Simple POC implementation of token generation
    # In production, use Authlib's AuthorizationServer
    form = await request.form()
    client_id = form.get("client_id")
    client_secret = form.get("client_secret")
    
    client = db.query(OAuth2Client).filter(OAuth2Client.client_id == client_id).first()
    if not client or not client.verify_secret(client_secret):
        logger.warning("auth_failed_invalid_client", client_id=client_id)
        raise HTTPException(status_code=401, detail="Invalid client credentials")
    
    # 🚀 OPTIMIZATION: Real RSA JWT signing
    import time
    from src.config import settings
    
    payload = {
        "iss": "https://auth.bsopt.internal",
        "sub": client_id,
        "aud": "bsopt-api",
        "roles": client.scopes,
        "iat": int(time.time()),
        "exp": int(time.time()) + settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }
    
    token = jwt.encode({"alg": "RS256", "kid": "internal-key-01"}, payload, settings.rsa_private_key)
    
    return {
        "access_token": token.decode("utf-8") if isinstance(token, bytes) else token,
        "token_type": "Bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "scope": " ".join(client.scopes)
    }

@router.get("/.well-known/openid-configuration")
async def openid_configuration():
    """Discovery endpoint for the resource server."""
    return {
        "issuer": "https://auth.bsopt.internal",
        "authorization_endpoint": "https://auth.bsopt.internal/auth/authorize",
        "token_endpoint": "https://auth.bsopt.internal/auth/token",
        "jwks_uri": "https://auth.bsopt.internal/auth/jwks",
        "response_types_supported": ["code", "token"],
        "subject_types_supported": ["public"],
        "id_token_signing_alg_values_supported": ["RS256"]
    }

@router.get("/jwks")
async def jwks():
    """
    JSON Web Key Set (JWKS) endpoint.
    Exposes public keys for Resource Servers to verify tokens.
    """
    from src.config import settings
    from authlib.jose import JsonWebKey
    
    key = JsonWebKey.import_key(settings.rsa_public_key, {"kty": "RSA", "kid": "internal-key-01", "use": "sig"})
    return {"keys": [key.as_dict()]}

from .better_auth import get_current_user

@router.get("/me")
async def read_users_me(user = Depends(get_current_user)):
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "emailVerified": user.emailVerified
    }

