import structlog
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

from src.config import settings

from .providers import OIDCProvider, auth_registry

logger = structlog.get_logger()

# 🚀 INITIALIZATION: Register providers
# Internal Auth Server (Neon-backed)
auth_registry.register(OIDCProvider(
    name="internal",
    issuer_url="https://auth.bsopt.internal", 
    audience="bsopt-api",
    public_key=settings.rsa_public_key
))

# Keycloak (Legacy support)
auth_registry.register(OIDCProvider(
    name="keycloak",
    issuer_url="http://auth-service:8080/realms/bsopt",
    audience="account"
))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="https://auth.bsopt.internal/protocol/openid-connect/token")

# 🚀 BACKWARD COMPATIBILITY: Shims for legacy tests
jwks_cache = {}

async def get_jwks():
    """Shim for legacy tests. Returns jwks from the keycloak provider."""
    provider = auth_registry.providers.get("keycloak")
    if provider:
        keys = await provider.get_jwks()
        if keys:
            jwks_cache["keys"] = keys.get("keys", [])
        return keys
    return None

async def verify_token(request: Request, token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Use registry to verify token from any registered issuer
        payload = await auth_registry.verify_any(token)
        
        # 🚀 FIX: Populate request.state.user for downstream dependencies
        class AuthenticatedUser:
            def __init__(self, payload):
                self.id = payload.get("sub")
                # Normalize roles across providers
                roles = payload.get("realm_access", {}).get("roles", []) or payload.get("roles", [])
                self.tier = "enterprise" if "admin" in roles else "free"
                self.email = payload.get("email")
        
        request.state.user = AuthenticatedUser(payload)
        return payload
    except Exception as e:
        logger.warning("token_verification_failed", error=str(e))
        raise credentials_exception

# Role-Based Access Control (RBAC) Dependency
class RoleChecker:
    def __init__(self, allowed_roles: list):
        self.allowed_roles = allowed_roles

    def __call__(self, token_payload: dict = Depends(verify_token)):
        # Normalize role check
        user_roles = token_payload.get("realm_access", {}).get("roles", []) or token_payload.get("roles", [])

        if not set(self.allowed_roles).intersection(user_roles):
            logger.warning("rbac_denied", user=token_payload.get("sub"), required=self.allowed_roles)
            raise HTTPException(status_code=403, detail="Insufficient Permissions")
        return token_payload
