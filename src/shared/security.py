import requests
import structlog
from typing import Dict, Any

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
            response = requests.post(self.opa_url, json=payload)
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

if __name__ == "__main__":
    enforcer = OPAEnforcer()
    user = {"id": "user123", "role": "trader"}
    print("Authorized:", enforcer.is_authorized(user, "read", "market_data"))
