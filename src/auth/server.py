from fastapi import APIRouter, Depends, HTTPException, Request, Form
from sqlalchemy.orm import Session
from src.database import get_db
from src.auth.service import AuthService, get_auth_service
import structlog

logger = structlog.get_logger()
router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/token")
async def generate_token(
    client_id: str = Form(...),
    client_secret: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    OAuth2 Token Endpoint.
    Validates client credentials and issues a JWT.
    """
    service = AuthService(db)
    client = service.verify_client(client_id, client_secret)
    
    if not client:
        logger.warning("auth_failed_invalid_client", client_id=client_id)
        raise HTTPException(status_code=401, detail="Invalid client credentials")
    
    token = service.create_token(client_id, client.scopes or [])
    
    return {
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": 30 * 60, # 30 mins
        "scope": " ".join(client.scopes or [])
    }

@router.get("/.well-known/openid-configuration")
async def openid_configuration(request: Request):
    """Discovery endpoint."""
    base_url = str(request.base_url).rstrip("/")
    return {
        "issuer": f"{base_url}/auth",
        "token_endpoint": f"{base_url}/auth/token",
        "jwks_uri": f"{base_url}/auth/jwks",
        "response_types_supported": ["token"],
        "subject_types_supported": ["public"],
        "id_token_signing_alg_values_supported": ["RS256"]
    }

@router.get("/jwks")
async def jwks():
    """Exposes public keys for token verification."""
    from src.config import settings
    from authlib.jose import JsonWebKey
    
    key = JsonWebKey.import_key(
        settings.rsa_public_key, 
        {"kty": "RSA", "kid": "internal-key-01", "use": "sig"}
    )
    return {"keys": [key.as_dict()]}