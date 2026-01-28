"""
HTTP Client Utility
===================

Provides a centralized, tuned, and persistent httpx.AsyncClient for service-to-service communication.
Ensures connection pooling and optimal timeout settings.
"""

import httpx
from typing import Optional
import structlog

logger = structlog.get_logger(__name__)

class HttpClientManager:
    _client: Optional[httpx.AsyncClient] = None

    @classmethod
    def get_client(cls) -> httpx.AsyncClient:
        """
        Get or create the singleton AsyncClient instance.
        """
        if cls._client is None:
            logger.info("initializing_shared_http_client")
            cls._client = httpx.AsyncClient(
                timeout=10.0,
                limits=httpx.Limits(
                    max_connections=100, 
                    max_keepalive_connections=20,
                    keepalive_expiry=30.0
                )
            )
        return cls._client

    @classmethod
    async def close(cls):
        """Close the shared client."""
        if cls._client:
            await cls._client.aclose()
            cls._client = None
            logger.info("closed_shared_http_client")

def get_http_client() -> httpx.AsyncClient:
    """Dependency injection helper."""
    return HttpClientManager.get_client()
