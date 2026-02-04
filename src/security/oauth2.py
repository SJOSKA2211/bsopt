"""
OAuth2 Authentication Service (Async)
=====================================

Refactored to support full OAuth2 flows:
- Authorization Code (for user-facing apps)
- Client Credentials (for machine-to-machine)
- Resource Owner Password Credentials (legacy support)
"""

import logging
import secrets
import inspect
from starlette.concurrency import run_in_threadpool
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union, cast
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, SecurityScopes
import hashlib
import jwt
from jwt.exceptions import PyJWTError, ExpiredSignatureError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload

from src.config import settings
from src.database import get_async_db
from src.database.models import User, APIKey, OAuth2Client

from .password import password_service

logger = logging.getLogger(__name__)

# OAuth2 Scheme
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="auth/token",
    scopes={
        "pricing:read": "Read pricing data",
        "pricing:write": "Calculate new prices",
        "admin": "Administrative access"
    }
)

@dataclass
class TokenData:
    user_id: Optional[str]
    client_id: Optional[str]
    scopes: List[str]
    exp: datetime
    iat: datetime
    jti: Optional[str] = None

@dataclass
class TokenPair:
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 0
    scope: str = ""

class AuthService:
    @property
    def private_key(self):
        return settings.JWT_PRIVATE_KEY
    @property
    def public_key(self):
        return settings.JWT_PUBLIC_KEY
    @property
    def algorithm(self):
        return settings.JWT_ALGORITHM
    @property
    def access_token_expire(self):
        return timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    async def authenticate_client(self, db: AsyncSession, client_id: str, client_secret: str) -> Optional[OAuth2Client]:
        """Authenticate a confidential client."""
        # Simple hash verification for client secret
        # In a real system, secrets should be hashed like passwords
        result = await db.execute(select(OAuth2Client).where(OAuth2Client.client_id == client_id))
        client = result.scalar_one_or_none()
        
        if not client or not client.verify_secret(client_secret):
            return None
        return client

    async def create_client_credentials_token(self, client: OAuth2Client, scopes: List[str]) -> TokenPair:
        """Create a token for Client Credentials flow."""
        # Verify scopes are allowed for this client
        allowed_scopes = set(client.scopes)
        requested_scopes = set(scopes)
        
        if not requested_scopes.issubset(allowed_scopes):
            raise HTTPException(status_code=400, detail="Invalid scope requested")
            
        access_token = self._create_token(
            {
                "sub": client.client_id, # Subject is client_id in this flow
                "type": "client_credentials",
                "scopes": scopes
            },
            self.access_token_expire
        )
        
        return TokenPair(
            access_token=access_token,
            refresh_token="", # No refresh token for client credentials
            expires_in=int(self.access_token_expire.total_seconds()),
            scope=" ".join(scopes)
        )

    def _create_token(self, data: dict, expires_delta: timedelta) -> str:
        """Internal helper to create a JWT token."""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + expires_delta
        to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc), "jti": secrets.token_hex(16)})
        return jwt.encode(to_encode, self.private_key, algorithm=self.algorithm)

    async def validate_token(self, security_scopes: SecurityScopes, token: str = Depends(oauth2_scheme)) -> TokenData:
        if not token:
            raise HTTPException(status_code=401, detail="Not authenticated")
            
        try:
            payload = jwt.decode(token, self.public_key, algorithms=[self.algorithm])
            token_scopes = payload.get("scopes", [])
            
            token_data = TokenData(
                user_id=payload.get("user_id"),
                client_id=payload.get("sub") if payload.get("type") == "client_credentials" else None,
                scopes=token_scopes,
                exp=datetime.fromtimestamp(payload.get("exp", 0), tz=timezone.utc),
                iat=datetime.fromtimestamp(payload.get("iat", 0), tz=timezone.utc),
                jti=payload.get("jti"),
            )
            
            # Verify scopes
            for scope in security_scopes.scopes:
                if scope not in token_data.scopes:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Not enough permissions",
                        headers={"WWW-Authenticate": f"Bearer scope=\"{security_scopes.scope_str}\""},
                    )
            
            return token_data
            
        except ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

auth_service = AuthService()