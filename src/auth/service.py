from typing import Optional
from sqlalchemy.orm import Session
from src.database.models import User, OAuth2Client
from src.config import settings
from authlib.jose import jwt, JoseError # Added JoseError
import time
import structlog
from fastapi import HTTPException # Added HTTPException

logger = structlog.get_logger(__name__)

class AuthService:
    """
    Core authentication and authorization service.
    Implements OAuth 2.0 / OpenID Connect logic for the BSOPT platform.
    """

    def __init__(self, db: Session):
        self.db = db

    def verify_client(self, client_id: str, client_secret: str) -> Optional[OAuth2Client]:
        """Verify client credentials."""
        client = self.db.query(OAuth2Client).filter(OAuth2Client.client_id == client_id).first()
        if client and client.verify_secret(client_secret):
            return client
        return None

    def create_token(self, client_id: str, scopes: list) -> str:
        """Issue a JWT for a verified client."""
        payload = {
            "iss": "https://auth.bsopt.internal",
            "sub": client_id,
            "aud": "bsopt-api",
            "roles": scopes,
            "iat": int(time.time()),
            "exp": int(time.time()) + settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
        
        # RSA signing using the private key from config
        token = jwt.encode(
            {"alg": settings.JWT_ALGORITHM, "kid": "internal-key-01"}, 
            payload, 
            settings.rsa_private_key
        )
        
        return token.decode("utf-8") if isinstance(token, bytes) else token

    def validate_token(self, token: str) -> dict:
        """Validate a JWT and return the payload."""
        try:
            payload = jwt.decode(token, settings.rsa_public_key)
            payload.validate()
            return payload
        except JoseError as e: # Catch specific JWT exceptions
            logger.warning("token_validation_failed_jwt_error", error=str(e))
            raise HTTPException(status_code=401, detail=f"Invalid or expired token: {e.__class__.__name__}")
        except Exception as e:
            logger.error("token_validation_failed_generic_error", error=str(e))
            raise HTTPException(status_code=500, detail=f"Authentication service error: {e.__class__.__name__}")

def get_auth_service(db: Session):
    return AuthService(db)
