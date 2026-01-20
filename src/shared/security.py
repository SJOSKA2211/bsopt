import requests
import structlog
from typing import Dict, Any, Optional
from fastapi import Request, HTTPException, status

logger = structlog.get_logger()

class OPAEnforcer:
    """
    Enforcer for Open Policy Agent (OPA) policies.
    Provides fine-grained authorization for the platform.
    """
    def __init__(self, opa_url: str = "http://localhost:8181/v1/data/authz/allow"):
        self.opa_url = opa_url

    def is_authorized(self, user: Dict[str, Any], action: str, resource: str) -> bool:
        """
        Query OPA to determine if the user is authorized for the given action and resource.
        """
        payload = {
            "input": {
                "user": user,
                "action": action,
                "resource": resource
            }
        }
        
        try:
            response = requests.post(self.opa_url, json=payload, timeout=2)
            if response.status_code == 200:
                result = response.json().get("result", False)
                logger.info("opa_authorization_check", 
                            user=user.get("id"), 
                            action=action, 
                            resource=resource, 
                            authorized=result)
                return result
            else:
                logger.error("opa_error", status_code=response.status_code)
                return False
        except Exception as e:
            logger.error("opa_connection_failed", error=str(e))
            return False

class MTLSVerifier:
    """
    Verifier for Mutual TLS (mTLS).
    In a zero-trust architecture, every service verifies the client certificate.
    """
    def __init__(self, required_dn: Optional[str] = None):
        self.required_dn = required_dn

    def verify(self, request: Request) -> bool:
        """
        Verify the client certificate from the request headers.
        Commonly passed by a reverse proxy like Nginx or Envoy.
        """
        # In a real mTLS setup, these headers are populated by the TLS terminator
        verify_status = request.headers.get("X-SSL-Client-Verify")
        client_dn = request.headers.get("X-SSL-Client-S-DN")
        
        if verify_status != "SUCCESS":
            logger.warning("mtls_verification_failed", status=verify_status)
            return False
        
        if self.required_dn and self.required_dn != client_dn:
            logger.warning("mtls_dn_mismatch", expected=self.required_dn, actual=client_dn)
            return False
            
        logger.info("mtls_verified", client_dn=client_dn)
        return True

# FastAPI Dependencies

async def verify_mtls(request: Request):
    """Dependency to enforce mTLS."""
    verifier = MTLSVerifier()
    if not verifier.verify(request):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="mTLS verification failed"
        )

def opa_authorize(action: str, resource: str):
    """Dependency to enforce OPA authorization."""
    async def _authorize(request: Request):
        # Extract user from JWT (assuming it was already validated)
        # For simplicity in this implementation, we look for a X-User-Role header
        # which would be populated by the API Gateway after JWT validation.
        user_id = request.headers.get("X-User-Id", "anonymous")
        user_role = request.headers.get("X-User-Role", "guest")
        user = {"id": user_id, "role": user_role}
        
        enforcer = OPAEnforcer()
        if not enforcer.is_authorized(user, action, resource):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"OPA Authorization failed for {action} on {resource}"
            )
    return _authorize

if __name__ == "__main__":
    enforcer = OPAEnforcer()
    user = {"id": "user123", "role": "trader"}
    print("Authorized:", enforcer.is_authorized(user, "read", "market_data"))
