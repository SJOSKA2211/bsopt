from typing import Dict
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from cachetools import TTLCache  # Optimization: Cache keys
import httpx
import structlog

logger = structlog.get_logger()

# Configuration
KEYCLOAK_URL = "http://auth-service:8080/realms/bsopt"
ALGORITHM = "RS256"
AUDIENCE = "account"

# 🚀 OPTIMIZATION: Cache JWKS for 1 hour to avoid network lag
# This keeps latency < 1ms for verification
jwks_cache = TTLCache(maxsize=1, ttl=3600)

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{KEYCLOAK_URL}/protocol/openid-connect/token"
)


async def get_jwks():
    if "keys" in jwks_cache:
        return jwks_cache["keys"]
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{KEYCLOAK_URL}/protocol/openid-connect/certs"
        )
        keys = resp.json()
        jwks_cache["keys"] = keys
        return keys


async def validate_token(token: str) -> Dict:
    """
    Validates the JWT token and returns the payload.
    Raises JWTError if validation fails.
    """
    try:
        # Get public keys
        jwks = await get_jwks()
        # Decode header to find which key signed this token
        unverified_header = jwt.get_unverified_header(token)
        rsa_key = {}
        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n": key["n"],
                    "e": key["e"]
                }
                break
        if not rsa_key:
            raise JWTError("Key not found")

        # Verify Signature
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=[ALGORITHM],
            audience=AUDIENCE,
            options={"verify_at_hash": False}
        )
        return payload
    except Exception as e:
        if not isinstance(e, JWTError):
            raise JWTError(str(e))
        raise e


async def verify_token(
    request: Request,
    token: str = Depends(oauth2_scheme)
) -> Dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = await validate_token(token)

        # 🚀 FIX: Populate request.state.user for downstream dependencies
        # (like opa_authorize)
        # We create a simple object-like wrapper or map the dictionary
        # to match the expected interface
        class AuthenticatedUser:
            def __init__(self, payload):
                self.id = payload.get("sub")
                # Map Keycloak roles to 'tier' for compatibility
                # with opa_authorize
                roles = payload.get("realm_access", {}).get("roles", [])
                self.tier = "enterprise" if "admin" in roles else "free"
                self.email = payload.get("email")

        request.state.user = AuthenticatedUser(payload)

        return payload  # Returns User Info (sub, roles, email)
    except JWTError:
        raise credentials_exception


# Role-Based Access Control (RBAC) Dependency
class RoleChecker:
    def __init__(self, allowed_roles: list):
        self.allowed_roles = allowed_roles

    def __call__(self, token_payload: Dict = Depends(verify_token)):
        # Extract roles from Keycloak token structure
        realm_access = token_payload.get("realm_access", {})
        user_roles = realm_access.get("roles", [])

        # Check intersection
        if not set(self.allowed_roles).intersection(user_roles):
            logger.warning(
                "rbac_denied",
                user=token_payload.get("sub"),
                required=self.allowed_roles
            )
            raise HTTPException(
                status_code=403,
                detail="Insufficient Permissions"
            )
        return token_payload
