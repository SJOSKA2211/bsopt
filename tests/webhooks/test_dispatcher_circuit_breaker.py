import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.utils.circuit_breaker import (
    CircuitBreaker,
)


# Placeholder for Celery task for red phase. Will be replaced by actual Celery task in next step
class MockDlqTask:
    def delay(self, *args, **kwargs):
        pass


@pytest.mark.asyncio
async def test_circuit_breaker_closed_to_open():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
    assert cb.is_closed
    assert not cb.is_open
    assert not cb.is_half_open

    for _ in range(cb.failure_threshold):
        await cb.record_failure()

    assert cb.is_open
    assert not cb.is_half_open


@pytest.mark.asyncio
async def test_circuit_breaker_open_to_half_open():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
    for _ in range(cb.failure_threshold):
        await cb.record_failure()
    assert cb.is_open

    await asyncio.sleep(cb.recovery_timeout + 0.1)
    assert cb.is_half_open
    assert not cb.is_open


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_success_to_closed():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1, expected_successes=1)
    for _ in range(cb.failure_threshold):
        await cb.record_failure()
    await asyncio.sleep(cb.recovery_timeout + 0.1)  # Half-open
    assert cb.is_half_open  # Ensure it's in half-open state

    await cb.record_success()
    assert cb.is_closed
    assert not cb.is_half_open


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_failure_to_open():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1, expected_successes=1)
    for _ in range(cb.failure_threshold):
        await cb.record_failure()
    await asyncio.sleep(cb.recovery_timeout + 0.1)  # Half-open
    assert cb.is_half_open  # Ensure it's in half-open state

    await cb.record_failure()
    assert cb.is_open
    assert not cb.is_half_open


@pytest.mark.asyncio
async def test_dispatcher_exponential_backoff_retries():
    mock_celery_app = AsyncMock()
    mock_dlq_task = MockDlqTask()  # Use mock dlq task
    cb = CircuitBreaker()

    dispatcher = WebhookDispatcher(
        celery_app=mock_celery_app, circuit_breaker=cb, dlq_task=mock_dlq_task
    )

    # Simulate network errors, then success
    mock_response_fail = AsyncMock(spec=httpx.Response)
    mock_response_fail.status_code = 500
    mock_response_fail.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500 Server Error",
        request=httpx.Request("POST", "http://test.com"),
        response=mock_response_fail,
    )

    mock_response_success = AsyncMock(spec=httpx.Response)
    mock_response_success.status_code = 200
    mock_response_success.raise_for_status.return_value = None

    with patch("httpx.AsyncClient") as MockAsyncClient:
        mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
        mock_client_instance.post.side_effect = [
            mock_response_fail,  # First attempt fails
            mock_response_fail,  # Second attempt fails
            mock_response_success,  # Third attempt succeeds
        ]

        url = "http://test.com/fail_then_success"
        payload = {"data": "test"}
        headers = {}
        secret = "secret"

        await dispatcher.dispatch_webhook(url, payload, headers, secret, retries=0)

        assert mock_client_instance.post.call_count == 3
        assert cb.is_closed  # Should reset after success


@pytest.mark.asyncio
async def test_dispatcher_dlq_routing():
    mock_celery_app = AsyncMock()
    mock_dlq_task = AsyncMock()  # Use AsyncMock for its delay method

    cb = CircuitBreaker(failure_threshold=1)  # Fails fast
    dispatcher = WebhookDispatcher(
        celery_app=mock_celery_app, circuit_breaker=cb, dlq_task=mock_dlq_task
    )

    # Simulate constant failure
    mock_response_fail = AsyncMock(spec=httpx.Response)
    mock_response_fail.status_code = 500
    mock_response_fail.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500 Server Error",
        request=httpx.Request("POST", "http://test.com"),
        response=mock_response_fail,
    )

    with patch("httpx.AsyncClient") as MockAsyncClient:
        mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
        mock_client_instance.post.side_effect = [
            mock_response_fail
        ] * 6  # 5 retries + 1 initial

        url = "http://test.com/fail_to_dlq"
        payload = {"data": "dlq"}
        headers = {}
        secret = "secret"

        await dispatcher.dispatch_webhook(url, payload, headers, secret, retries=0)

        # Should call send_to_dlq_task
        mock_dlq_task.delay.assert_called_once()
        args, kwargs = mock_dlq_task.delay.call_args
        assert args[0]["url"] == url
        assert args[0]["payload"] == payload
        assert "circuit_breaker_open" in args[0]["reason"]


@pytest.mark.asyncio
async def test_dispatcher_circuit_breaker_skipped():
    mock_celery_app = AsyncMock()
    mock_dlq_task = AsyncMock()
    cb = CircuitBreaker(failure_threshold=1)

    # Force circuit breaker open
    await cb.record_failure()
    await cb.record_failure()  # Go to open state

    dispatcher = WebhookDispatcher(
        celery_app=mock_celery_app, circuit_breaker=cb, dlq_task=mock_dlq_task
    )

    with patch("httpx.AsyncClient") as MockAsyncClient:
        mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
        mock_client_instance.post.new_callable = AsyncMock  # Ensure it's async

        url = "http://test.com/skipped"
        payload = {"data": "skipped"}
        headers = {}
        secret = "secret"

        await dispatcher.dispatch_webhook(url, payload, headers, secret)

        mock_client_instance.post.assert_not_called()
        mock_dlq_task.delay.assert_called_once()
        args, kwargs = mock_dlq_task.delay.call_args
        assert "circuit_breaker_open" in args[0]["reason"]


@pytest.mark.asyncio
async def test_dispatcher_request_error_retry():
    mock_celery_app = AsyncMock()
    mock_dlq_task = AsyncMock()
    cb = CircuitBreaker()

    dispatcher = WebhookDispatcher(
        celery_app=mock_celery_app, circuit_breaker=cb, dlq_task=mock_dlq_task
    )

    # Simulate network errors, then success
    with patch("httpx.AsyncClient") as MockAsyncClient:
        mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
        mock_client_instance.post.side_effect = [
            httpx.RequestError(
                "Network error", request=httpx.Request("POST", "http://test.com")
            ),
            httpx.RequestError(
                "Network error", request=httpx.Request("POST", "http://test.com")
            ),
            AsyncMock(status_code=200),  # Third attempt succeeds
        ]

        url = "http://test.com/request_error_then_success"
        payload = {"data": "test"}
        headers = {}
        secret = "secret"

        await dispatcher.dispatch_webhook(url, payload, headers, secret, retries=0)

        assert mock_client_instance.post.call_count == 3
        assert cb.is_closed  # Should reset after success


@pytest.mark.asyncio
async def test_dispatcher_dlq_on_unexpected_exception():
    mock_celery_app = AsyncMock()
    mock_dlq_task = AsyncMock()
    cb = CircuitBreaker()

    dispatcher = WebhookDispatcher(
        celery_app=mock_celery_app, circuit_breaker=cb, dlq_task=mock_dlq_task
    )

    # Simulate unexpected exception
    with patch("httpx.AsyncClient") as MockAsyncClient:
        mock_client_instance = MockAsyncClient.return_value.__aenter__.return_value
        mock_client_instance.post.side_effect = Exception("Unexpected Error")

        url = "http://test.com/unexpected_error"
        payload = {"data": "error"}
        headers = {}
        secret = "secret"

        await dispatcher.dispatch_webhook(url, payload, headers, secret)

        mock_dlq_task.delay.assert_called_once()
        args, kwargs = mock_dlq_task.delay.call_args
        assert "unexpected_error" in args[0]["reason"]
