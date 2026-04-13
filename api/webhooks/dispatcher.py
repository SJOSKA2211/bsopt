from src.shared.webhooks.dispatcher import (
    WebhookDispatcher as SharedDispatcher,
)
from src.shared.webhooks.dispatcher import (
    generate_signature,
    verify_signature,
)


async def _generate_signature(secret: str, payload: str, timestamp: int | None = None) -> str:
    """Shim for legacy signature generation."""
    return await generate_signature(secret, payload, timestamp)


async def _verify_signature(
    secret: str, payload: str, timestamp: int, signature: str, tolerance: int = 300
) -> bool:
    """Shim for legacy signature verification."""
    return await verify_signature(secret, payload, timestamp, signature, tolerance)


class WebhookDispatcher(SharedDispatcher):
    """Legacy wrapper for API compatibility."""

    pass