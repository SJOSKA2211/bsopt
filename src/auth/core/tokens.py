"""
Asymmetric JWT Token Substrate (RS256/ES256).
"""

import logging
import secrets
from datetime import UTC, datetime, timedelta

import jwt
import msgspec
from jwt.exceptions import ExpiredSignatureError, PyJWTError
from pydantic import BaseModel

from src.shared.config import settings

logger = logging.getLogger(__name__)


class TokenData(msgspec.Struct, frozen=True):
    """Standardized Token Data (msgspec)."""

    user_id: str
    email: str
    tier: str
    token_type: str
    exp: datetime
    iat: datetime
    jti: str | None = None
    scopes: list[str] = []


class TokenPair(BaseModel):
    """Standardized Token Pair (Pydantic V2)."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    requires_mfa: bool = False


class TokenService:
    """
    JWT asymmetric token management.
    """

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

        to_encode = {
            **data,
            "exp": expire,
            "iat": now,
            "jti": secrets.token_hex(24),
            "iss": "manifold-auth-v2",
        }

        # Use msgspec for fast serialization check (ensures no non-JSON serializable objects)
        msgspec.json.encode(to_encode)

        algorithm = settings.JWT_ALGORITHM
        key = self._get_key_for_algorithm(algorithm, is_private=True)
        return jwt.encode(to_encode, key, algorithm=algorithm)

    def decode_token(self, token: str) -> TokenData:
        """Decode and validate a JWT token."""
        try:
            # 🛡️ Sentinel: Fix JWT Algorithm Confusion Vulnerability
            # Never trust the `alg` header from unverified token payload.
            # Always enforce the algorithm defined in settings to prevent algorithm confusion attacks.
            algorithm = settings.JWT_ALGORITHM
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
            from fastapi import HTTPException

            raise HTTPException(status_code=401, detail="Token has expired")
        except PyJWTError:
            from fastapi import HTTPException

            raise HTTPException(status_code=401, detail="Invalid token")


# Global instance for easy access
token_service = TokenService()
