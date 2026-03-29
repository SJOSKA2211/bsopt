import hashlib
import hmac
import time
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from src.api.webhooks.dispatcher import (
    WebhookDispatcher,
    _generate_signature,
    _verify_signature,
)


@pytest.mark.asyncio
async def test_signature_generation_and_verification():
    secret = "super_secret"
    payload = {"event": "test", "data": 123}
    payload_str = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS).decode("utf-8")

    # Generate
    sig_header = await _generate_signature(secret, payload_str)
    assert sig_header.startswith("t=")
    assert ",sha256=" in sig_header

    # Parse parts
    t_part, s_part = sig_header.split(",")
    timestamp = int(t_part.split("=")[1])
    signature = s_part.split("=")[1]

    # Verify valid
    assert await _verify_signature(secret, payload_str, timestamp, signature)

    # Verify invalid secret
    assert not await _verify_signature("wrong_secret", payload_str, timestamp, signature)

    # Verify invalid payload
    assert not await _verify_signature(secret, "wrong_payload", timestamp, signature)

@pytest.mark.asyncio
async def test_signature_timestamp_tolerance():
    secret = "secret"
    payload = "data"
    now = int(time.time())

    # Valid (within 5 mins)
    sig = hmac.new(secret.encode(), f"{now}.{payload}".encode(), hashlib.sha256).hexdigest()
    assert await _verify_signature(secret, payload, now, sig)

    # Invalid (too old)
    old_ts = now - 600
    sig_old = hmac.new(secret.encode(), f"{old_ts}.{payload}".encode(), hashlib.sha256).hexdigest()
    assert not await _verify_signature(secret, payload, old_ts, sig_old)

@pytest.mark.asyncio
async def test_dispatch_webhook_success():
    mock_cb = MagicMock(side_effect=lambda f: f)  # Bypass decorator
    mock_client = AsyncMock()
    mock_client.post.return_value = MagicMock(status_code=200)

    with patch("src.shared.http_client.HttpClientManager.get_client", return_value=mock_client):
        dispatcher = WebhookDispatcher(celery_app=None, circuit_breaker=mock_cb, dlq_task=None)
        await dispatcher.dispatch_webhook("http://example.com", {"a": 1}, {}, "secret")

        mock_client.post.assert_called_once()
        args, kwargs = mock_client.post.call_args
        assert kwargs["headers"]["X-Webhook-Signature"].startswith("t=")

@pytest.mark.asyncio
async def test_dispatch_webhook_circuit_breaker_open():
    # Simulate a circuit breaker that raises when called
    def cb_decorator(func):
        async def wrapper(*args, **kwargs):
            raise Exception("Circuit Breaker is OPEN")

        return wrapper

    dispatcher = WebhookDispatcher(celery_app=None, circuit_breaker=cb_decorator, dlq_task=None)

    with pytest.raises(Exception) as exc:
        await dispatcher.dispatch_webhook("http://example.com", {}, {}, "secret")

    assert "OPEN" in str(exc.value)

@pytest.mark.asyncio
async def test_dispatch_webhook_general_failure():
    mock_cb = MagicMock(side_effect=lambda f: f)
    mock_client = AsyncMock()
    mock_client.post.side_effect = Exception("HTTP Error")

    with patch("src.shared.http_client.HttpClientManager.get_client", return_value=mock_client):
        dispatcher = WebhookDispatcher(celery_app=None, circuit_breaker=mock_cb, dlq_task=None)
        with pytest.raises(Exception) as exc:
            await dispatcher.dispatch_webhook("http://example.com", {}, {}, "secret")
        assert "HTTP Error" in str(exc.value)
