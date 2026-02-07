"""
HTTP Client Utility
===================

Provides a centralized, tuned, and persistent httpx.AsyncClient for service-to-service communication.
Ensures connection pooling and optimal timeout settings.
"""

import httpx
import structlog

logger = structlog.get_logger(__name__)

import asyncio

import structlog

logger = structlog.get_logger(__name__)


class HttpClientManager:
    _client: httpx.AsyncClient | None = None
    _semaphore: asyncio.Semaphore | None = None

    @classmethod
    def get_client(cls) -> httpx.AsyncClient:
        """
        Get or create the singleton AsyncClient instance with SOTA pooling.
        """
        if cls._client is None:
            logger.info("initializing_shared_http_client_c100k")
            # 🚀 SINGULARITY: High-concurrency limits and HTTP/2 multiplexing
            cls._client = httpx.AsyncClient(
                http2=True,
                timeout=httpx.Timeout(10.0, connect=5.0),
                limits=httpx.Limits(
                    max_connections=5000,
                    max_keepalive_connections=1000,
                    keepalive_expiry=60.0,
                ),
            )
        return cls._client

    @classmethod
    def get_semaphore(cls, limit: int = 20) -> asyncio.Semaphore:
        """Get or create a concurrency semaphore."""
        if cls._semaphore is None:
            cls._semaphore = asyncio.Semaphore(limit)
        return cls._semaphore

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
