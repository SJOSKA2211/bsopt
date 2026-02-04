from fastapi import APIRouter, Depends, HTTPException, Request, Form
from sqlalchemy.orm import Session
from src.database import get_db
from src.database.models import User, OAuth2Client, OAuth2Token, OAuth2AuthorizationCode
from src.config import settings
from authlib.integrations.sqla_oauth2 import (
    create_query_client_func,
    create_save_token_func,
)
from authlib.oauth2 import AuthorizationServer
from authlib.oauth2.rfc6749 import grants
from authlib.oauth2.rfc7636 import CodeChallenge
import structlog
import time

logger = structlog.get_logger()
router = APIRouter(prefix="/auth", tags=["auth"])

# --- AUTHLIB INTEGRATION ---

query_client = create_query_client_func(Session, OAuth2Client)
save_token = create_save_token_func(Session, OAuth2Token)

class AuthorizationCodeGrant(grants.AuthorizationCodeGrant):
    def save_authorization_code(self, code, request):
        auth_code = OAuth2AuthorizationCode(
            code=code,
            client_id=request.client.client_id,
            redirect_uri=request.redirect_uri,
            scope=request.scope,
            user_id=request.user.id,
            code_challenge=request.data.get('code_challenge'),
            code_challenge_method=request.data.get('code_challenge_method'),
        )
        self.db.add(auth_code)
        self.db.commit()
        return auth_code

    def query_authorization_code(self, code, client):
        item = self.db.query(OAuth2AuthorizationCode).filter_by(code=code, client_id=client.client_id).first()
        if item and not item.is_expired():
            return item

    def delete_authorization_code(self, authorization_code):
        self.db.delete(authorization_code)
        self.db.commit()

    def authenticate_user(self, authorization_code):
        return self.db.get(User, authorization_code.user_id)

server = AuthorizationServer(query_client, save_token)
server.register_grant(AuthorizationCodeGrant, [CodeChallenge(required=True)])

# --- ENDPOINTS ---

@router.post("/authorize")
async def authorize(request: Request, db: Session = Depends(get_db)):
    """
    Authorization Endpoint.
    In a full implementation, this would handle the login UI and user consent.
    For this high-performance API, we expect a pre-authenticated session or credential.
    """
    # Placeholder for user authentication logic
    user = db.query(User).first() # Dummy for now, would be replaced by real session/user
    if not user:
        raise HTTPException(status_code=401, detail="User not authenticated")
    
    return server.create_authorization_response(request, user)

@router.post("/token")
async def token(request: Request):
    """
    Token Endpoint.
    Handles token exchange for Auth Code and Refresh Tokens with PKCE support.
    """
    return server.create_token_response(request)

@router.get("/jwks")
async def jwks():
    """Exposes public keys for token verification."""
    from authlib.jose import JsonWebKey
    
    key = JsonWebKey.import_key(
        settings.rsa_public_key, 
        {"kty": "RSA", "kid": "internal-key-01", "use": "sig"}
    )
    return {"keys": [key.as_dict()]}

@router.get("/.well-known/openid-configuration")
async def openid_configuration(request: Request):
    """Discovery endpoint for OIDC."""
    base_url = str(request.base_url).rstrip("/")
    return {
        "issuer": f"{base_url}/auth",
        "authorization_endpoint": f"{base_url}/auth/authorize",
        "token_endpoint": f"{base_url}/auth/token",
        "jwks_uri": f"{base_url}/auth/jwks",
        "response_types_supported": ["code", "token"],
        "subject_types_supported": ["public"],
        "id_token_signing_alg_values_supported": ["RS256"],
        "code_challenge_methods_supported": ["S256"]
    }