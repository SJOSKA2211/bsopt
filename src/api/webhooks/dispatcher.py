from src.shared.webhooks.dispatcher import WebhookDispatcher as SharedDispatcher


class WebhookDispatcher(SharedDispatcher):
    """Legacy wrapper for API compatibility."""
    pass

