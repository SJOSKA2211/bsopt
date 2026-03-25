import time
from collections.abc import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.auth.auth import auth_service
from src.shared.security import SecurityContext

logger = structlog.get_logger(__name__)

class ZeroTrustAuthMiddleware(BaseHTTPMiddleware):
    """
    Production-Grade Zero-Trust Middleware.
    Intercepts every request to validate Asymmetric JWTs and enforce mTLS headers.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip auth for docs, health, and auth endpoints
        path = request.url.path
        if path.startswith(("/docs", "/redoc", "/openapi.json", "/auth/login", "/auth/register", "/health")):
            return await call_next(request)

        start_time = time.perf_counter()
        
        # 1. Extract Token
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return Response(content="Missing or invalid authorization header", status_code=401)
        
        token = auth_header.split(" ")[1]

        try:
            # 2. Validate Token (Uses Redis cache + Asymmetric Cryptography)
            token_data = await auth_service.validate_token(token)
            
            # 3. Inject Security Context
            request.state.security_context = SecurityContext(
                user_id=token_data.user_id,
                email=token_data.email,
                tier=token_data.tier,
                auth_type="jwt",
                is_internal=False
            )
            
            # Compatibility with legacy code
            request.state.user_id = token_data.user_id
            request.state.user_tier = token_data.tier

        except Exception as e:
            logger.warning("middleware_auth_failed", path=path, error=str(e))
            return Response(content="Unauthorized", status_code=401)

        # 4. mTLS Header Enforcement (Forwarded by Envoy)
        if not self._verify_mtls(request):
            return Response(content="Mutual TLS required", status_code=403)

        response = await call_next(request)
        
        process_time = (time.perf_counter() - start_time) * 1000
        logger.info("request_authenticated", path=path, latency_ms=f"{process_time:.2f}ms")
        
        return response

    def _verify_mtls(self, request: Request) -> bool:
        """Verify mTLS status if in production."""
        from src.shared.config import settings
        if not settings.is_production:
            return True
            
        return request.headers.get("X-SSL-Client-Verify") == "SUCCESS"
