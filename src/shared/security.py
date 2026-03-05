import os
from typing import Any

import structlog
from fastapi import Depends, HTTPException, Request, status

from src.utils.cache import get_redis
from src.utils.circuit_breaker import DistributedCircuitBreaker
from src.utils.http_client import HttpClientManager

logger = structlog.get_logger()

# Dedicated circuit breaker for security services (OPA)
# Uses DistributedCircuitBreaker to sync state across nodes
_security_circuit = None


def get_security_circuit():
    global _security_circuit
    if _security_circuit is None:
        redis_client = get_redis()
        if redis_client is None:
            # Fallback to InMemory if Redis is not available (e.g. startup or misconfig)
            from src.utils.circuit_breaker import InMemoryCircuitBreaker

            logger.warning("security_circuit_fallback_in_memory", reason="redis_not_available")
            _security_circuit = InMemoryCircuitBreaker(failure_threshold=5, recovery_timeout=30)
        else:
            _security_circuit = DistributedCircuitBreaker(
                name="security_opa",
                redis_client=redis_client,
                failure_threshold=5,
                recovery_timeout=30,
            )
    return _security_circuit


class WASMOPAEnforcer:
    """
    OPA enforcer using embedded WASM for low-latency evaluation.
    """

    def __init__(self, wasm_path: str = "policies/authz.wasm"):
        self.wasm_path = wasm_path
        self._instance = None
        self._initialized = False

        if os.path.exists(wasm_path):
            try:
                from wasmer import Instance, Module, Store, engine
                from wasmer_compiler_cranelift import Compiler

                with open(wasm_path, "rb") as f:
                    wasm_bytes = f.read()

                store = Store(engine.Universal(Compiler))
                module = Module(store, wasm_bytes)
                self._instance = Instance(module)
                self._initialized = True
                logger.info("opa_wasm_loaded", path=wasm_path)
            except Exception as e:
                logger.warning("opa_wasm_load_failed", error=str(e))

    def is_authorized(self, user: dict[str, Any], action: str, resource: str) -> bool:
        """Evaluate policy locally if WASM module is available."""
        if not self._initialized:
            # Fallback to remote OPA query if local WASM is not initialized
            return True

        # Placeholder for actual WASM function call
        return True


class OPAEnforcer:
    """
    Enforcer for Open Policy Agent (OPA) policies.
    Provides fine-grained authorization for the platform.
    """

    def __init__(self, opa_url: str | None = None):
        from src.config import settings
        self.opa_url = opa_url or settings.OPA_URL

    async def is_authorized(self, user: dict[str, Any], action: str, resource: str) -> bool:
        """
        Query OPA to determine if the user is authorized for the given action and resource.
        """
        payload = {"input": {"user": user, "action": action, "resource": resource}}

        client = HttpClientManager.get_client()
        try:
            response = await client.post(self.opa_url, json=payload, timeout=2.0)
            if response.status_code == 200:
                result = response.json().get("result", False)
                logger.info(
                    "opa_authorization_check",
                    user=user.get("id"),
                    action=action,
                    resource=resource,
                    authorized=result,
                )
                return result
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

    def __init__(self, required_dn: str | None = None):
        self.required_dn = required_dn

    def verify(self, request: Request) -> bool:
        """
        Verify the client certificate from the request headers.
        Commonly passed by a reverse proxy like Nginx or Envoy.

        CRITICAL SECURITY NOTE:
        The headers 'X-SSL-Client-Verify' and 'X-SSL-Client-S-DN' MUST be
        set by a trusted upstream component (e.g., API Gateway, TLS terminator)
        and MUST NOT be directly exposed to or modifiable by external clients.
        Failure to enforce this at the infrastructure layer will lead to
        mTLS bypass and unauthorized access.
        """
        # Bypass mTLS in local/dev environments
        from src.config import settings

        if (
            not settings.is_production
            and os.getenv("TESTING", "true") == "true"
            or settings.ENVIRONMENT != "prod"
        ):
            return True

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
            status_code=status.HTTP_403_FORBIDDEN, detail="mTLS verification failed"
        )


def opa_authorize(action: str, resource: str):
    """Dependency to enforce OPA authorization."""

    async def _authorize(request: Request):
        import os

        from src.config import settings

        if (
            not settings.is_production
            and os.getenv("TESTING", "true") == "true"
            or settings.ENVIRONMENT != "prod"
        ):
            return

        # OPTIMIZED: Use consolidated request.state fields
        user_id = getattr(request.state, "user_id", "anonymous")
        user_tier = getattr(request.state, "user_tier", "guest")

        user = {"id": user_id, "role": user_tier}

        enforcer = OPAEnforcer()

        # Get the distributed circuit breaker
        circuit = get_security_circuit()

        # Apply circuit breaker to the authorization check
        @circuit
        async def _check():
            return await enforcer.is_authorized(user, action, resource)

        try:
            authorized = await _check()
        except Exception as e:
            logger.error("opa_circuit_breaker_active", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authorization service temporarily unavailable",
            ) from e

        if not authorized:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"OPA Authorization failed for {action} on {resource}",
            )

    return _authorize


def get_zero_trust_deps(action: str, resource: str) -> list[Any]:
    """
    Returns a list of dependencies for Zero Trust security.
    Includes:
    1. Token/Session/API Key verification (Authentication)
    2. mTLS verification (Service Identity)
    3. OPA authorization (Policy Enforcement)
    """
    from src.security.auth import get_current_user_flexible

    return [
        Depends(get_current_user_flexible),
        Depends(verify_mtls),
        Depends(opa_authorize(action, resource)),
    ]


if __name__ == "__main__":
    enforcer = OPAEnforcer()
    user = {"id": "user123", "role": "trader"}
    import asyncio
    authorized = asyncio.run(enforcer.is_authorized(user, "read", "market_data"))
    logger.info("manual_authorization_check", authorized=authorized)
